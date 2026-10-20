import numpy as np
import os
from icecream import ic

import openmdao.api as om
from openaerostruct.utils.constants import grav_constant


def norm(vec):
    return np.sqrt(np.sum(vec**2))


class FuelLoads(om.ExplicitComponent):
    """
    Compute the nodal loads from the distributed fuel within the wing
    to be applied to the wing structure.
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        self.surface = surface = self.options["surface"]
        self.ny = surface["mesh"].shape[1]

        self.add_input("fuel_vols", val=np.ones((self.ny - 1)), units="m**3")
        self.add_input("nodes", val=np.zeros((self.ny, 3)), units="m")
        self.add_input("fuel_mass", val=1.0, units="kg")
        self.add_input("load_factor", val=1.0)
        self.add_input(
            "wing_fuel_fraction",
            val=1.0,
            desc="Fraction which specifies the amount of fuel that the wings carry. 1.0 denotes all fuel carried in the wings. Intent is to use this to specify the amount of fuel that a center tank may carry for the wings.",
        )

        self.add_output("fuel_weight_loads", val=np.zeros((self.ny, 6)), units="N")

        self.declare_partials("fuel_weight_loads", "fuel_vols")
        self.declare_partials("fuel_weight_loads", "nodes")
        self.declare_partials("fuel_weight_loads", "fuel_mass")
        self.declare_partials("fuel_weight_loads", "load_factor")
        self.declare_partials("fuel_weight_loads", "wing_fuel_fraction")

    def compute(self, inputs, outputs):
        nodes = inputs["nodes"]

        element_lengths = np.ones(self.ny - 1, dtype=complex)
        for i in range(self.ny - 1):
            element_lengths[i] = norm(nodes[i + 1] - nodes[i])

        # And we also need the deltas between consecutive nodes
        deltas = nodes[1:, :] - nodes[:-1, :]

        # Fuel weight
        fuel_weight = (
            (inputs["fuel_mass"] + self.surface["Wf_reserve"])
            * grav_constant
            * inputs["load_factor"]
        )

        # Modify the fuel weight depending on the wing fuel fraction (if 1.0, all fuel in wings; if wing_fuel_fraction < 1, then (1-wing_fuel_fraction) amount of fuel is stored in central tank)
        wing_fuel_frac = inputs["wing_fuel_fraction"]
        fuel_weight = fuel_weight * wing_fuel_frac

        # print(f"\nfuel_loads wing_fuel_fraction = {wing_fuel_frac}")

        if self.surface["symmetry"]:
            fuel_weight /= 2.0

        vols = inputs["fuel_vols"]
        sum_vols = np.sum(vols)

        # Now we need the fuel weight per segment
        # Assume it's divided evenly based on vols
        z_weights = vols * fuel_weight / sum_vols

        # Assume weight coincides with the elastic axis
        z_forces_for_each = z_weights / 2.0
        z_moments_for_each = (
            z_weights
            * element_lengths
            / 12.0
            * (deltas[:, 0] ** 2 + deltas[:, 1] ** 2) ** 0.5
            / element_lengths
        )

        loads = np.zeros((self.ny, 6), dtype=complex)

        # Loads in z-direction
        loads[:-1, 2] = loads[:-1, 2] - z_forces_for_each
        loads[1:, 2] = loads[1:, 2] - z_forces_for_each

        # Bending moments for consistency
        loads[:-1, 3] = (
            loads[:-1, 3] - z_moments_for_each * deltas[:, 1] / element_lengths
        )
        loads[1:, 3] = (
            loads[1:, 3] + z_moments_for_each * deltas[:, 1] / element_lengths
        )

        loads[:-1, 4] = (
            loads[:-1, 4] - z_moments_for_each * deltas[:, 0] / element_lengths
        )
        loads[1:, 4] = (
            loads[1:, 4] + z_moments_for_each * deltas[:, 0] / element_lengths
        )

        outputs["fuel_weight_loads"] = loads

        if os.environ.get("FLUME_DEBUG_AERO") == "1":
            print(
                "[FuelLoads] surface={} fuel_mass={:.6e} load_factor={:.6e} "
                "sum_fuel_vols={:.6e} fuel_load_norm={:.6e}".format(
                    self.surface["name"],
                    np.real(inputs["fuel_mass"]).item(),
                    np.real(inputs["load_factor"]).item(),
                    float(np.real(sum_vols)),
                    float(np.linalg.norm(np.real(loads))),
                ),
                flush=True,
            )

    def compute_partials(self, inputs, partials):
        ny = self.ny
        nodes = inputs["nodes"]
        vols = inputs["fuel_vols"]
        fuel_mass = float(inputs["fuel_mass"][0])
        load_factor = float(inputs["load_factor"][0])

        reserve = self.surface["Wf_reserve"]
        sym = 0.5 if self.surface["symmetry"] else 1.0

        # Total distributed fuel weight and its sensitivities to the scalars
        FW = (fuel_mass + reserve) * grav_constant * load_factor * sym

        # Modify the fuel weight depending on the wing fuel fraction (if 1.0, all fuel in wings; if wing_fuel_fraction < 1, then (1-wing_fuel_fraction) amount of fuel is stored in central tank)
        wing_fuel_frac = float(inputs["wing_fuel_fraction"][0])
        FW = FW * wing_fuel_frac

        # These sensitivities must include the wing_fuel_fraction factor, since
        # the distributed fuel weight (and hence every load) is scaled by it in
        # compute(). dFW_dwff is the derivative wrt the fraction itself, so it
        # must NOT carry the factor (FW is linear in the fraction).
        dFW_dm = grav_constant * load_factor * sym * wing_fuel_frac
        dFW_dlf = (fuel_mass + reserve) * grav_constant * sym * wing_fuel_frac
        dFW_dwff = FW / wing_fuel_frac

        S = np.sum(vols)
        zw = vols * FW / S  # per-element fuel weight

        # Element geometry
        deltas = nodes[1:, :] - nodes[:-1, :]
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dz = deltas[:, 2]
        hl = np.sqrt(dx**2 + dy**2)  # in-plane (x-y) length
        L = np.sqrt(dx**2 + dy**2 + dz**2)  # full element length

        # The element_lengths cancel in the moment terms, leaving
        #   m3_i = zw_i * C3_i,   m4_i = zw_i * C4_i,
        # with P = hl / (12 L),  C3 = P*dy,  C4 = P*dx.
        P = hl / (12.0 * L)
        C3 = P * dy
        C4 = P * dx

        # Derivatives of P wrt the element deltas
        Pdx = (1.0 / 12.0) * dx * dz**2 / (hl * L**3)
        Pdy = (1.0 / 12.0) * dy * dz**2 / (hl * L**3)
        Pdz = -(1.0 / 12.0) * hl * dz / L**3

        # dC3/d(dx,dy,dz) and dC4/d(dx,dy,dz) via the product rule
        dC3 = np.stack([Pdx * dy, Pdy * dy + P, Pdz * dy], axis=1)  # (ny-1, 3)
        dC4 = np.stack([Pdx * dx + P, Pdy * dx, Pdz * dx], axis=1)  # (ny-1, 3)

        nout = 6 * ny

        # --- wrt nodes (only the moment dofs 3 and 4 depend on nodes) ---
        Jn = np.zeros((nout, 3 * ny))
        for i in range(ny - 1):
            for c in range(3):
                A3 = zw[i] * dC3[i, c]  # d(m3_i)/d(nodes[i+1, c])
                A4 = zw[i] * dC4[i, c]
                ci = i * 3 + c
                cip1 = (i + 1) * 3 + c
                r3i, r3ip1 = i * 6 + 3, (i + 1) * 6 + 3
                r4i, r4ip1 = i * 6 + 4, (i + 1) * 6 + 4
                # loads[i, 3] = -m3_i ; loads[i+1, 3] = +m3_i
                Jn[r3i, cip1] += -A3
                Jn[r3i, ci] += A3
                Jn[r3ip1, cip1] += A3
                Jn[r3ip1, ci] += -A3
                # loads[i, 4] = -m4_i ; loads[i+1, 4] = +m4_i
                Jn[r4i, cip1] += -A4
                Jn[r4i, ci] += A4
                Jn[r4ip1, cip1] += A4
                Jn[r4ip1, ci] += -A4
        partials["fuel_weight_loads", "nodes"] = Jn

        # Helper to scatter a per-element dzw into the load Jacobian columns.
        def scatter(dzw):
            col = np.zeros(nout)
            for i in range(ny - 1):
                dzf = 0.5 * dzw[i]
                dm3 = C3[i] * dzw[i]
                dm4 = C4[i] * dzw[i]
                # forces (dof 2): loads[i,2] -= zf ; loads[i+1,2] -= zf
                col[i * 6 + 2] += -dzf
                col[(i + 1) * 6 + 2] += -dzf
                # moment dof 3: loads[i,3] -= m3 ; loads[i+1,3] += m3
                col[i * 6 + 3] += -dm3
                col[(i + 1) * 6 + 3] += dm3
                # moment dof 4: loads[i,4] -= m4 ; loads[i+1,4] += m4
                col[i * 6 + 4] += -dm4
                col[(i + 1) * 6 + 4] += dm4
            return col

        # --- wrt fuel_mass and load_factor (scalars) ---
        dzw_dm = (vols / S) * dFW_dm
        dzw_dlf = (vols / S) * dFW_dlf
        dzw_dwff = (vols / S) * dFW_dwff

        partials["fuel_weight_loads", "fuel_mass"] = scatter(dzw_dm).reshape(nout, 1)
        partials["fuel_weight_loads", "load_factor"] = scatter(dzw_dlf).reshape(nout, 1)
        partials["fuel_weight_loads", "wing_fuel_fraction"] = scatter(dzw_dwff).reshape(
            nout, 1
        )

        # --- wrt fuel_vols ---
        # zw_i = v_i * FW / S  =>  d(zw_i)/d(v_k) = FW * (delta_ik / S - v_i / S**2)
        Jv = np.zeros((nout, ny - 1))
        for k in range(ny - 1):
            dzw_dvk = FW * (-vols / S**2)
            dzw_dvk[k] += FW / S
            Jv[:, k] = scatter(dzw_dvk)
        partials["fuel_weight_loads", "fuel_vols"] = Jv
