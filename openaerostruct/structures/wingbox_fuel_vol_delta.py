import numpy as np

import openmdao.api as om
from icecream import ic


def norm(vec):
    return np.sqrt(np.sum(vec**2))


class WingboxFuelVolDelta(om.ExplicitComponent):
    """
    Create a constraint to ensure the wingbox has enough internal volume to
    store the required fuel.

    Parameters
    ----------
    fuelburn : float
        Fuel weight
    fuel_vols[ny-1] : numpy array
        The magnitude of each individual panel's fuel-carrying volumes.

    Returns
    -------
    fuel_vol_delta : float
        If the value is negative, then there isn't enough volume for the fuel.
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        self.surface = surface = self.options["surface"]

        self.ny = surface["mesh"].shape[1]

        self.add_input("fuelburn", val=0.0, units="kg")
        self.add_input("fuel_vols", val=np.zeros((self.ny - 1)), units="m**3")
        self.add_input(
            "wing_fuel_fraction",
            val=1.0,
            desc="Fraction which specifies the amount of fuel that the wings carry. 1.0 denotes all fuel carried in the wings. Intent is to use this to specify the amount of fuel that a center tank may carry for the wings.",
        )

        self.add_output("fuel_vol_delta", val=0.0, units="m**3")

        # Analytic partials.
        #   fuel_vol_delta = sum(fuel_vols)
        #                    - sym * (fuelburn + reserves) * wing_fuel_fraction / fuel_density
        # d/d(fuel_vols) = 1 is a true constant. The fuelburn and
        # wing_fuel_fraction partials both depend on wing_fuel_fraction (and,
        # for the latter, on fuelburn), so they are set in compute_partials.
        fuel_density = surface["fuel_density"]
        self.fuel_density = fuel_density
        self.sym_factor = 0.5 if surface["symmetry"] is True else 1.0

        self.declare_partials(
            "fuel_vol_delta", "fuel_vols", val=np.ones((1, self.ny - 1))
        )
        self.declare_partials("fuel_vol_delta", "fuelburn")
        self.declare_partials("fuel_vol_delta", "wing_fuel_fraction")

    def compute(self, inputs, outputs):
        reserves = self.surface["Wf_reserve"]
        fuel_density = self.surface["fuel_density"]
        vols = inputs["fuel_vols"]

        # Use non-in-place arithmetic so the input vector is not mutated
        # (mutating inputs corrupts complex-step checks and repeated evals).
        fuelburn = inputs["fuelburn"]

        # Compute the total weight of the fuel (fuelburn + reserves)
        total_fuel_weight = fuelburn + reserves

        # Compute the percentage of the weight that is stored in the wings by multiplying by wing_fuel_frac
        wing_fuel_frac = inputs["wing_fuel_fraction"]

        # print(f"\nwingbox_fuel_vol_delta wing_fuel_fraction = {wing_fuel_frac}")

        wing_fuel_weight = total_fuel_weight * wing_fuel_frac

        if self.surface["symmetry"] is True:
            # fuel_weight = fuel_weight / 2.0
            # reserves = reserves / 2.0
            wing_fuel_weight = wing_fuel_weight / 2.0

        # Compute the amount of fuel that is stored in the wings by multiplying (fuel_weight + reserves) by wing_fuel_frac

        sum_vols = np.sum(vols)

        print("\nVolume summary: ")
        print(f"\tVolume in one wingbox = {sum_vols:.4e} m^3")

        # fuel_vol = (fuel_weight + reserves) / fuel_density
        wing_fuel_vol = wing_fuel_weight / fuel_density
        wing_fuel_vol = wing_fuel_vol.item()
        print(
            f"\tTotal vol of fuel (fuelburn + reserves) * wing_fuel_frac / 2 = {wing_fuel_vol:.4e} m^3"
        )
        ratio = wing_fuel_vol / sum_vols
        print(f"\tRatio of wing_fuel_vol to wingbox vol = {ratio:.3f}")

        fuel_vol_delta = sum_vols - wing_fuel_vol
        print(f"\tfuel_vol_delta = {fuel_vol_delta:.3f} m^3")

        # This is used for the fuel-volume constraint. It should be positive for fuel to fit.
        outputs["fuel_vol_delta"] = fuel_vol_delta
        # outputs["fuel_vol_delta"] = sum_vols - (fuel_weight + reserves) / fuel_density

    def compute_partials(self, inputs, partials):
        reserves = self.surface["Wf_reserve"]
        wing_fuel_frac = inputs["wing_fuel_fraction"]
        fuelburn = inputs["fuelburn"]

        # fuel_vol_delta = sum(fuel_vols)
        #                  - sym * (fuelburn + reserves) * wing_fuel_fraction / fuel_density
        partials["fuel_vol_delta", "fuelburn"] = (
            -self.sym_factor * wing_fuel_frac / self.fuel_density
        )
        partials["fuel_vol_delta", "wing_fuel_fraction"] = (
            -self.sym_factor * (fuelburn + reserves) / self.fuel_density
        )
