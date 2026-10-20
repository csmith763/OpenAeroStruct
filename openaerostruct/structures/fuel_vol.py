import numpy as np

import openmdao.api as om


def norm(vec):
    return np.sqrt(np.sum(vec**2))


class WingboxFuelVol(om.ExplicitComponent):
    """
    Computes the internal volumes of the wingbox segments.

    Parameters
    ----------
    nodes[ny, 3] : numpy array
        Coordinates of FEM nodes.
    A_int[ny-1] : numpy array
        Internal cross-sectional area of each wingbox segment.

    Returns
    -------
    fuel_vols[ny-1] : numpy array
        Internal volume of each wingbox segment.
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        self.surface = surface = self.options["surface"]

        self.ny = surface["mesh"].shape[1]

        self.add_input("nodes", val=np.zeros((self.ny, 3)), units="m")
        self.add_input("A_int", val=np.zeros((self.ny - 1)), units="m**2")
        self.add_output("fuel_vols", val=np.zeros((self.ny - 1)), units="m**3")

        ny = self.ny

        # fuel_vols[i] = L[i] * A_int[i], with L[i] = ||nodes[i+1] - nodes[i]||.
        # d(fuel_vols)/d(A_int) is diagonal (each segment depends only on its
        # own A_int).
        arange = np.arange(ny - 1)
        self.declare_partials("fuel_vols", "A_int", rows=arange, cols=arange)

        # fuel_vols[i] depends on nodes[i] and nodes[i+1] (3 coords each), so the
        # Jacobian wrt the flattened (ny, 3) nodes has two 3-entry blocks per row.
        rows = np.repeat(arange, 6)
        cols = np.empty(6 * (ny - 1), dtype=int)
        for i in range(ny - 1):
            # columns for nodes[i, :] then nodes[i+1, :]
            cols[6 * i : 6 * i + 3] = 3 * i + np.arange(3)
            cols[6 * i + 3 : 6 * i + 6] = 3 * (i + 1) + np.arange(3)
        self.declare_partials("fuel_vols", "nodes", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        nodes = inputs["nodes"]

        element_lengths = np.zeros(self.ny - 1, dtype=type(nodes[0, 0]))

        for i in range(self.ny - 1):
            element_lengths[i] = norm(nodes[i + 1] - nodes[i])

        # Next we multiply the element lengths with the A_int for the internal volumes of the wingobox segments
        vols = element_lengths * inputs["A_int"]

        outputs["fuel_vols"] = vols

    def compute_partials(self, inputs, partials):
        nodes = inputs["nodes"]
        A_int = inputs["A_int"]

        # Deltas and lengths for each element
        deltas = nodes[1:, :] - nodes[:-1, :]  # (ny-1, 3)
        element_lengths = np.sqrt(np.sum(deltas**2, axis=1))  # (ny-1,)

        # d(fuel_vols[i])/d(A_int[i]) = L[i]
        partials["fuel_vols", "A_int"] = element_lengths

        # d(L[i])/d(nodes[i+1, c]) =  deltas[i, c] / L[i]
        # d(L[i])/d(nodes[i,   c]) = -deltas[i, c] / L[i]
        # d(fuel_vols[i])/d(nodes) = A_int[i] * d(L[i])/d(nodes)
        dL = deltas / element_lengths[:, np.newaxis]  # (ny-1, 3)
        scaled = A_int[:, np.newaxis] * dL  # (ny-1, 3)

        jac = np.empty(6 * (self.ny - 1))
        for i in range(self.ny - 1):
            jac[6 * i : 6 * i + 3] = -scaled[i]  # wrt nodes[i, :]
            jac[6 * i + 3 : 6 * i + 6] = scaled[i]  # wrt nodes[i+1, :]
        partials["fuel_vols", "nodes"] = jac
