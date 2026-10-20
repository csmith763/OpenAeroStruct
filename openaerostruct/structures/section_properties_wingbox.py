import numpy as np

import openmdao.api as om


class SectionPropertiesWingbox(om.ExplicitComponent):
    """
    Compute geometric cross-section properties for the wingbox elements.
    See Chauhan et al. (https://doi.org/10.1007/978-3-319-97773-7_38) for more.

    Parameters
    ----------
    streamwise_chords[ny-1] : numpy array
        Average streamwise chord lengths for each streamwise VLM panel.
    fem_chords[ny-1] : numpy array
        Effective chord lengths normal to the FEM elements.
    fem_twists[ny-1] : numpy array
        Twist angles in planes normal to the FEM elements.
    spar_thickness[ny-1] : numpy array
        Material thicknesses of the front and rear spars for each wingbox segment.
    skin_thickness[ny-1] : numpy array
        Material thicknesses of the top and bottom skins for each wingbox segment.
    t_over_c[ny-1] : numpy array
        Streamwise thickness-to-chord ratios for each wingbox segment.


    Returns
    -------
    A[ny-1] : numpy array
        Cross-sectional area of each wingbox segment.
    A_enc[ny-1] : numpy array
        Cross-sectional enclosed area (measured using the material midlines) of
        each wingbox segment.
    A_int[ny-1] : numpy array
        Cross-sectional internal area of each wingbox segment (used for fuel
        volume).
    Iy[ny-1] : numpy array
        Second moment of area about the neutral axis parallel to the local
        y-axis (for each wingbox segment).
    Qz[ny-1] : numpy array
        First moment of area above the neutral axis parallel to the local
        z-axis (for each wingbox segment).
    Iz[ny-1] : numpy array
        Second moment of area about the neutral axis parallel to the local
        z-axis (for each wingbox segment).
    J[ny-1] : numpy array
        Torsion constants for each wingbox segment.
    htop[ny-1] : numpy array
        Distance to the point on the top skin that is the farthest away from
        the local-z neutral axis (for each wingbox segment).
    hbottom[ny-1] : numpy array
        Distance to the point on the bottom skin that is the farthest away from
        the local-z neutral axis (for each wingbox segment).
    hfront[ny-1] : numpy array
        Distance to the point on the front spar that is the farthest away from
        the local-y neutral axis (for each wingbox segment).
    hrear[ny-1] : numpy array
        Distance to the point on the rear spar that is the farthest away
        from the local-y neutral axis (for each wingbox segment).
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        self.surface = surface = self.options["surface"]

        self.mesh = surface["mesh"]
        self.ny = self.mesh.shape[1]

        # original thickness-to-chord ratio of the airfoil provided by the user
        self.orig_wb_af_t_over_c = surface["original_wingbox_airfoil_t_over_c"]

        # airfoil coordinates provided by the user
        self.data_x_upper = surface["data_x_upper"]
        self.data_x_lower = surface["data_x_lower"]
        self.data_y_upper = surface["data_y_upper"]
        self.data_y_lower = surface["data_y_lower"]

        self.add_input("streamwise_chords", val=np.ones((self.ny - 1)), units="m")
        self.add_input("fem_chords", val=np.ones((self.ny - 1)), units="m")
        self.add_input("fem_twists", val=np.ones((self.ny - 1)), units="deg")

        self.add_input("spar_thickness", val=np.ones((self.ny - 1)), units="m")
        self.add_input("skin_thickness", val=np.ones((self.ny - 1)), units="m")
        self.add_input("t_over_c", val=np.ones((self.ny - 1)))

        self.add_output("A", val=np.ones((self.ny - 1)), units="m**2")
        self.add_output("A_enc", val=np.ones((self.ny - 1)), units="m**2")
        self.add_output("A_int", val=np.ones((self.ny - 1)), units="m**2")
        self.add_output("Iy", val=np.ones((self.ny - 1)), units="m**4")
        self.add_output("Qz", val=np.ones((self.ny - 1)), units="m**3")
        self.add_output("Iz", val=np.ones((self.ny - 1)), units="m**4")
        self.add_output("J", val=np.ones((self.ny - 1)), units="m**4")
        self.add_output("htop", val=np.ones((self.ny - 1)), units="m")
        self.add_output("hbottom", val=np.ones((self.ny - 1)), units="m")
        self.add_output("hfront", val=np.ones((self.ny - 1)), units="m")
        self.add_output("hrear", val=np.ones((self.ny - 1)), units="m")

        # Every output segment depends only on the same-index input segment, so
        # all Jacobians are diagonal.
        arange = np.arange(self.ny - 1)
        outs = ["A", "A_enc", "A_int", "Iy", "Qz", "Iz", "J", "htop", "hbottom", "hfront", "hrear"]
        ins = [
            "streamwise_chords",
            "fem_chords",
            "fem_twists",
            "spar_thickness",
            "skin_thickness",
            "t_over_c",
        ]
        for o in outs:
            for j in ins:
                self.declare_partials(o, j, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        # NOTE: In the code below, the x- and y-axes correspond to the element
        # local z- and y-axes, respectively.

        chord = inputs["fem_chords"]
        spar_thickness = inputs["spar_thickness"]
        skin_thickness = inputs["skin_thickness"]
        t_over_c_original = self.orig_wb_af_t_over_c
        t_over_c = inputs["t_over_c"]
        streamwise_chord = inputs["streamwise_chords"]
        theta = inputs["fem_twists"]

        # Scale data points with chord
        data_x_upper = np.outer(self.data_x_upper, chord)
        data_y_upper = np.outer(self.data_y_upper, chord)
        data_x_lower = np.outer(self.data_x_lower, chord)
        data_y_lower = np.outer(self.data_y_lower, chord)

        # Scale y-coordinates by t/c design variable which is streamwise t/c
        data_y_upper *= t_over_c / t_over_c_original * streamwise_chord / chord
        data_y_lower *= t_over_c / t_over_c_original * streamwise_chord / chord

        # Compute enclosed area for torsion constant
        # This currently does not change with twist
        # Also compute internal area for internal volume calculation for fuel
        x_up_diff = data_x_upper[1:] - data_x_upper[:-1]
        x_low_diff = data_x_lower[1:] - data_x_lower[:-1]
        y_up_diff = data_y_upper[1:] - data_y_upper[:-1]
        y_low_diff = data_y_lower[1:] - data_y_lower[:-1]

        y_up_add = data_y_upper[1:] + data_y_upper[:-1]
        y_low_add = data_y_lower[1:] + data_y_lower[:-1]

        A_enc = (x_up_diff) * (y_up_add - skin_thickness) / 2  # area above 0 line
        A_enc += (x_low_diff) * (-y_low_add - skin_thickness) / 2  # area below 0 line
        A_int = (x_up_diff) * (y_up_add - 2 * skin_thickness) / 2  # area above 0 line
        A_int += (x_low_diff) * (-y_low_add - 2 * skin_thickness) / 2  # area below 0 line

        A_enc = np.sum(A_enc, axis=0)
        A_int = np.sum(A_int, axis=0)

        A_enc -= (data_y_upper[0] - data_y_lower[0]) * spar_thickness / 2  # area of spars
        A_enc -= (data_y_upper[-1] - data_y_lower[-1]) * spar_thickness / 2  # area of spars
        A_int -= (data_y_upper[0] - data_y_lower[0]) * spar_thickness  # area of spars
        A_int -= (data_y_upper[-1] - data_y_lower[-1]) * spar_thickness  # area of spars

        outputs["A_enc"] = A_enc
        outputs["A_int"] = A_int

        # Compute perimeter to thickness ratio for torsion constant
        # This currently does not change with twist
        p_by_t_1 = (x_up_diff**2 + y_up_diff**2) ** 0.5 / skin_thickness  # length / thickness of caps
        p_by_t_2 = (x_low_diff**2 + y_low_diff**2) ** 0.5 / skin_thickness  # length / thickness of caps

        p_by_t = np.sum(p_by_t_1 + p_by_t_2, axis=0)

        p_by_t += (data_y_upper[0] - data_y_lower[0] - skin_thickness) / spar_thickness  # length / thickness of spars
        p_by_t += (data_y_upper[-1] - data_y_lower[-1] - skin_thickness) / spar_thickness  # length / thickness of spars

        # Torsion constant
        J = 4 * A_enc**2 / p_by_t

        outputs["J"] = J

        # Rotate the wingbox
        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        data_x_upper_2 = rot_mat[0, 0] * data_x_upper + rot_mat[0, 1] * data_y_upper
        data_y_upper_2 = rot_mat[1, 0] * data_x_upper + rot_mat[1, 1] * data_y_upper
        data_x_lower_2 = rot_mat[0, 0] * data_x_lower + rot_mat[0, 1] * data_y_lower
        data_y_lower_2 = rot_mat[1, 0] * data_x_lower + rot_mat[1, 1] * data_y_lower

        data_x_upper = data_x_upper_2.copy()
        data_y_upper = data_y_upper_2.copy()
        data_x_lower = data_x_lower_2.copy()
        data_y_lower = data_y_lower_2.copy()

        x_up_diff = data_x_upper[1:] - data_x_upper[:-1]
        x_low_diff = data_x_lower[1:] - data_x_lower[:-1]
        y_up_diff = data_y_upper[1:] - data_y_upper[:-1]
        y_low_diff = data_y_lower[1:] - data_y_lower[:-1]

        y_up_add = data_y_upper[1:] + data_y_upper[:-1]
        y_low_add = data_y_lower[1:] + data_y_lower[:-1]

        # Compute area moment of inertia about x axis
        # First compute centroid and area
        first_moment_area_upper = (y_up_add / 2 - (skin_thickness / 2)) * skin_thickness * x_up_diff
        upper_area = skin_thickness * x_up_diff

        first_moment_area_lower = (y_low_add / 2 + (skin_thickness / 2)) * skin_thickness * x_low_diff
        lower_area = skin_thickness * x_low_diff

        first_moment_area_front_spar = (
            (data_y_upper[0] - data_y_lower[0] - 2 * skin_thickness)
            * spar_thickness
            * (data_y_upper[0] + data_y_lower[0])
            / 2
        )
        first_moment_area_rear_spar = (
            (data_y_upper[-1] - data_y_lower[-1] - 2 * skin_thickness)
            * spar_thickness
            * (data_y_upper[-1] + data_y_lower[-1])
            / 2
        )
        area_spars = (
            (data_y_upper[0] - data_y_lower[0] - 2 * skin_thickness)
            + (data_y_upper[-1] - data_y_lower[-1] - 2 * skin_thickness)
        ) * spar_thickness

        area = np.sum(upper_area, axis=0) + np.sum(lower_area, axis=0) + area_spars

        outputs["A"] = area

        centroid = (
            np.sum(first_moment_area_upper, axis=0)
            + np.sum(first_moment_area_lower, axis=0)
            + first_moment_area_front_spar
            + first_moment_area_rear_spar
        ) / area

        # Then compute area moment of inertia for upward bending
        # This is calculated using derived analytical expression assuming linear interpolation between airfoil data points
        a = y_up_diff / x_up_diff
        b = (y_up_diff + skin_thickness) / 2
        x2 = x_up_diff

        I_horiz_1 = 2 * (
            1.0 / 12.0 * a**3 * x2**4
            + 1.0 / 3.0 * a**2 * x2**3 * b
            + 1.0 / 2.0 * a * x2**2 * b**2
            + 1.0 / 3.0 * b**3 * x2
        )
        I_horiz_2 = x2 * skin_thickness * (y_up_add / 2 - skin_thickness / 2 - centroid) ** 2

        I_horiz = np.sum(I_horiz_1 + I_horiz_2, axis=0)

        # Compute area moment of inertia about y axis
        a = -y_low_diff / x_low_diff
        b = (-y_low_diff + skin_thickness) / 2
        x2 = x_low_diff

        I_horiz += np.sum(
            2
            * (
                1.0 / 12.0 * a**3 * x2**4
                + 1.0 / 3.0 * a**2 * x2**3 * b
                + 1.0 / 2.0 * a * x2**2 * b**2
                + 1.0 / 3.0 * b**3 * x2
            ),
            axis=0,
        )
        I_horiz += np.sum(x2 * skin_thickness * ((-y_low_add) / 2 - skin_thickness / 2 + centroid) ** 2, axis=0)

        # Contribution from the forward spar
        I_horiz += (
            1.0 / 12.0 * spar_thickness * (data_y_upper[0] - data_y_lower[0] - 2 * skin_thickness) ** 3
            + spar_thickness
            * (data_y_upper[0] - data_y_lower[0] - 2 * skin_thickness)
            * ((data_y_upper[0] + data_y_lower[0]) / 2 - centroid) ** 2
        )
        # Contribution from the rear spar
        I_horiz += (
            1.0 / 12.0 * spar_thickness * (data_y_upper[-1] - data_y_lower[-1] - 2 * skin_thickness) ** 3
            + spar_thickness
            * (data_y_upper[-1] - data_y_lower[-1] - 2 * skin_thickness)
            * ((data_y_upper[-1] + data_y_lower[-1]) / 2 - centroid) ** 2
        )

        outputs["Iz"] = I_horiz

        # Compute the Q required for transverse shear stress due to upward bending
        Q_upper = np.sum(((y_up_add / 2 - (skin_thickness / 2)) - centroid) * skin_thickness * x_up_diff, axis=0)

        Q_upper += (data_y_upper[0] - skin_thickness - centroid) ** 2 / 2 * (spar_thickness)
        Q_upper += (data_y_upper[-1] - skin_thickness - centroid) ** 2 / 2 * (spar_thickness)

        outputs["Qz"] = Q_upper

        # Compute area moment of inertia for backward bending
        I_vert = 0
        first_moment_area_front = (
            (data_y_upper[0] - data_y_lower[0]) * spar_thickness * (data_x_upper[0] + spar_thickness / 2)
        )
        first_moment_area_rear = (
            (data_y_upper[-1] - data_y_lower[-1]) * spar_thickness * (data_x_upper[-1] - spar_thickness / 2)
        )
        centroid_Ivert = (first_moment_area_front + first_moment_area_rear) / (
            ((data_y_upper[0] - data_y_lower[0]) + (data_y_upper[-1] - data_y_lower[-1])) * spar_thickness
        )

        I_vert += (
            1.0 / 12.0 * (data_y_upper[0] - data_y_lower[0]) * spar_thickness**3
            + (data_y_upper[0] - data_y_lower[0])
            * spar_thickness
            * (centroid_Ivert - (data_x_upper[0] + spar_thickness / 2)) ** 2
        )
        I_vert += (
            1.0 / 12.0 * (data_y_upper[-1] - data_y_lower[-1]) * spar_thickness**3
            + (data_y_upper[-1] - data_y_lower[-1])
            * spar_thickness
            * (data_x_upper[-1] - spar_thickness / 2 - centroid_Ivert) ** 2
        )

        # Add contribution of skins
        I_vert += 2 * (
            1.0 / 12.0 * skin_thickness * (data_x_upper[-1] - data_x_upper[0] - 2 * spar_thickness) ** 3
            + skin_thickness
            * (data_x_upper[-1] - data_x_upper[0] - 2 * spar_thickness)
            * (centroid_Ivert - (data_x_upper[-1] + data_x_upper[0]) / 2) ** 2
        )

        outputs["Iy"] = I_vert

        # Distances for calculating max bending stresses (KS function used)
        ks_rho = 500.0  # Hard coded, see Martins and Poon 2005 for more
        fmax_upper = np.max(data_y_upper, axis=0)
        htop = fmax_upper + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (data_y_upper - fmax_upper)), axis=0)) - centroid

        fmax_lower = np.max(-data_y_lower, axis=0)
        hbottom = (
            fmax_lower + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (-data_y_lower - fmax_lower)), axis=0)) + centroid
        )

        hfront = centroid_Ivert - data_x_upper[0]
        hrear = data_x_upper[-1] - centroid_Ivert

        outputs["htop"] = htop
        outputs["hbottom"] = hbottom
        outputs["hfront"] = hfront
        outputs["hrear"] = hrear

    def compute_partials(self, inputs, partials):
        # Analytic partials via forward-mode (tangent) propagation. Each output
        # segment depends only on the same-index input segment, so the Jacobians
        # are diagonal. We propagate a unit "seed" through the exact sequence of
        # operations in ``compute`` once per input, giving the diagonal entries.
        toc_orig = self.orig_wb_af_t_over_c
        Xu0 = np.real(self.data_x_upper)
        Xl0 = np.real(self.data_x_lower)
        Yu0 = np.real(self.data_y_upper)
        Yl0 = np.real(self.data_y_lower)

        chord = inputs["fem_chords"]
        st = inputs["spar_thickness"]
        sk = inputs["skin_thickness"]
        toc = inputs["t_over_c"]
        swc = inputs["streamwise_chords"]
        theta = inputs["fem_twists"]

        ks_rho = 500.0

        def _cubic_d(a, d_a, b, d_b, x2, d_x2):
            # derivative of 2*(a^3 x2^4/12 + a^2 x2^3 b/3 + a x2^2 b^2/2 + b^3 x2/3)
            return 2.0 * (
                1.0 / 12.0 * (3.0 * a**2 * d_a * x2**4 + a**3 * 4.0 * x2**3 * d_x2)
                + 1.0 / 3.0 * (2.0 * a * d_a * x2**3 * b + a**2 * 3.0 * x2**2 * d_x2 * b + a**2 * x2**3 * d_b)
                + 1.0 / 2.0 * (d_a * x2**2 * b**2 + a * 2.0 * x2 * d_x2 * b**2 + a * x2**2 * 2.0 * b * d_b)
                + 1.0 / 3.0 * (3.0 * b**2 * d_b * x2 + b**3 * d_x2)
            )

        def fwd(dchord, dst, dsk, dtoc, dswc, dtheta):
            # ----- scaled coordinates (pre-rotation) -----
            sc = swc / chord
            d_sc = (dswc * chord - swc * dchord) / chord**2
            scaleY = (toc / toc_orig) * sc
            d_scaleY = (dtoc / toc_orig) * sc + (toc / toc_orig) * d_sc

            Xu = np.outer(Xu0, chord)
            d_Xu = np.outer(Xu0, dchord)
            Xl = np.outer(Xl0, chord)
            d_Xl = np.outer(Xl0, dchord)
            bYu = np.outer(Yu0, chord)
            d_bYu = np.outer(Yu0, dchord)
            bYl = np.outer(Yl0, chord)
            d_bYl = np.outer(Yl0, dchord)
            Yu = bYu * scaleY
            d_Yu = d_bYu * scaleY + bYu * d_scaleY
            Yl = bYl * scaleY
            d_Yl = d_bYl * scaleY + bYl * d_scaleY

            # ----- A_enc, A_int, p_by_t, J (pre-rotation) -----
            xud, d_xud = Xu[1:] - Xu[:-1], d_Xu[1:] - d_Xu[:-1]
            xld, d_xld = Xl[1:] - Xl[:-1], d_Xl[1:] - d_Xl[:-1]
            yud, d_yud = Yu[1:] - Yu[:-1], d_Yu[1:] - d_Yu[:-1]
            yld, d_yld = Yl[1:] - Yl[:-1], d_Yl[1:] - d_Yl[:-1]
            yua, d_yua = Yu[1:] + Yu[:-1], d_Yu[1:] + d_Yu[:-1]
            yla, d_yla = Yl[1:] + Yl[:-1], d_Yl[1:] + d_Yl[:-1]

            d_Aenc = np.sum(d_xud * (yua - sk) / 2 + xud * (d_yua - dsk) / 2, axis=0) + np.sum(
                d_xld * (-yla - sk) / 2 + xld * (-d_yla - dsk) / 2, axis=0
            )
            d_Aint = np.sum(d_xud * (yua - 2 * sk) / 2 + xud * (d_yua - 2 * dsk) / 2, axis=0) + np.sum(
                d_xld * (-yla - 2 * sk) / 2 + xld * (-d_yla - 2 * dsk) / 2, axis=0
            )
            # spar area removals
            d_Aenc = d_Aenc - ((d_Yu[0] - d_Yl[0]) * st + (Yu[0] - Yl[0]) * dst) / 2
            d_Aenc = d_Aenc - ((d_Yu[-1] - d_Yl[-1]) * st + (Yu[-1] - Yl[-1]) * dst) / 2
            d_Aint = d_Aint - ((d_Yu[0] - d_Yl[0]) * st + (Yu[0] - Yl[0]) * dst)
            d_Aint = d_Aint - ((d_Yu[-1] - d_Yl[-1]) * st + (Yu[-1] - Yl[-1]) * dst)

            # values needed for J
            Aenc = np.sum(xud * (yua - sk) / 2, axis=0) + np.sum(xld * (-yla - sk) / 2, axis=0)
            Aenc = Aenc - (Yu[0] - Yl[0]) * st / 2 - (Yu[-1] - Yl[-1]) * st / 2

            s1 = np.sqrt(xud**2 + yud**2)
            d_s1 = (xud * d_xud + yud * d_yud) / s1
            s2 = np.sqrt(xld**2 + yld**2)
            d_s2 = (xld * d_xld + yld * d_yld) / s2
            pbt = np.sum(s1 / sk + s2 / sk, axis=0)
            d_pbt = np.sum((d_s1 * sk - s1 * dsk) / sk**2 + (d_s2 * sk - s2 * dsk) / sk**2, axis=0)
            pbt = pbt + (Yu[0] - Yl[0] - sk) / st + (Yu[-1] - Yl[-1] - sk) / st
            d_pbt = d_pbt + ((d_Yu[0] - d_Yl[0] - dsk) * st - (Yu[0] - Yl[0] - sk) * dst) / st**2
            d_pbt = d_pbt + ((d_Yu[-1] - d_Yl[-1] - dsk) * st - (Yu[-1] - Yl[-1] - sk) * dst) / st**2

            d_J = 4.0 * (2.0 * Aenc * d_Aenc * pbt - Aenc**2 * d_pbt) / pbt**2

            # ----- rotation by fem_twists (matches compute: np.cos/np.sin of raw value) -----
            cos = np.cos(theta)
            sin = np.sin(theta)
            dcos = -sin * dtheta
            dsin = cos * dtheta

            RXu = cos * Xu + sin * Yu
            d_RXu = dcos * Xu + cos * d_Xu + dsin * Yu + sin * d_Yu
            RYu = -sin * Xu + cos * Yu
            d_RYu = -dsin * Xu - sin * d_Xu + dcos * Yu + cos * d_Yu
            RXl = cos * Xl + sin * Yl
            d_RXl = dcos * Xl + cos * d_Xl + dsin * Yl + sin * d_Yl
            RYl = -sin * Xl + cos * Yl
            d_RYl = -dsin * Xl - sin * d_Xl + dcos * Yl + cos * d_Yl

            Xud, d_Xud = RXu[1:] - RXu[:-1], d_RXu[1:] - d_RXu[:-1]
            Xld, d_Xld = RXl[1:] - RXl[:-1], d_RXl[1:] - d_RXl[:-1]
            Yud, d_Yud = RYu[1:] - RYu[:-1], d_RYu[1:] - d_RYu[:-1]
            Yld, d_Yld = RYl[1:] - RYl[:-1], d_RYl[1:] - d_RYl[:-1]
            Yua, d_Yua = RYu[1:] + RYu[:-1], d_RYu[1:] + d_RYu[:-1]
            Yla, d_Yla = RYl[1:] + RYl[:-1], d_RYl[1:] + d_RYl[:-1]

            # ----- area and centroid -----
            uarea = sk * Xud
            d_uarea = dsk * Xud + sk * d_Xud
            larea = sk * Xld
            d_larea = dsk * Xld + sk * d_Xld
            hfs = RYu[0] - RYl[0] - 2 * sk
            d_hfs = d_RYu[0] - d_RYl[0] - 2 * dsk
            hrs = RYu[-1] - RYl[-1] - 2 * sk
            d_hrs = d_RYu[-1] - d_RYl[-1] - 2 * dsk
            area_spars = (hfs + hrs) * st
            d_area_spars = (d_hfs + d_hrs) * st + (hfs + hrs) * dst
            area = np.sum(uarea, axis=0) + np.sum(larea, axis=0) + area_spars
            d_area = np.sum(d_uarea, axis=0) + np.sum(d_larea, axis=0) + d_area_spars

            fmau = (Yua / 2 - sk / 2) * sk * Xud
            d_fmau = (
                (d_Yua / 2 - dsk / 2) * sk * Xud
                + (Yua / 2 - sk / 2) * dsk * Xud
                + (Yua / 2 - sk / 2) * sk * d_Xud
            )
            fmal = (Yla / 2 + sk / 2) * sk * Xld
            d_fmal = (
                (d_Yla / 2 + dsk / 2) * sk * Xld
                + (Yla / 2 + sk / 2) * dsk * Xld
                + (Yla / 2 + sk / 2) * sk * d_Xld
            )
            cf0 = (RYu[0] + RYl[0]) / 2
            d_cf0 = (d_RYu[0] + d_RYl[0]) / 2
            cr0 = (RYu[-1] + RYl[-1]) / 2
            d_cr0 = (d_RYu[-1] + d_RYl[-1]) / 2
            fmafs = hfs * st * cf0
            d_fmafs = (d_hfs * st + hfs * dst) * cf0 + hfs * st * d_cf0
            fmars = hrs * st * cr0
            d_fmars = (d_hrs * st + hrs * dst) * cr0 + hrs * st * d_cr0

            num_c = np.sum(fmau, axis=0) + np.sum(fmal, axis=0) + fmafs + fmars
            d_num_c = np.sum(d_fmau, axis=0) + np.sum(d_fmal, axis=0) + d_fmafs + d_fmars
            centroid = num_c / area
            d_centroid = (d_num_c * area - num_c * d_area) / area**2

            # ----- Iz (upward bending) -----
            a = Yud / Xud
            d_a = (d_Yud * Xud - Yud * d_Xud) / Xud**2
            b = (Yud + sk) / 2
            d_b = (d_Yud + dsk) / 2
            d_Iz = np.sum(_cubic_d(a, d_a, b, d_b, Xud, d_Xud), axis=0)
            t = Yua / 2 - sk / 2 - centroid
            d_t = d_Yua / 2 - dsk / 2 - d_centroid
            d_Iz = d_Iz + np.sum(
                d_Xud * sk * t**2 + Xud * dsk * t**2 + Xud * sk * 2 * t * d_t, axis=0
            )
            # lower
            a = -Yld / Xld
            d_a = -(d_Yld * Xld - Yld * d_Xld) / Xld**2
            b = (-Yld + sk) / 2
            d_b = (-d_Yld + dsk) / 2
            d_Iz = d_Iz + np.sum(_cubic_d(a, d_a, b, d_b, Xld, d_Xld), axis=0)
            tl = -Yla / 2 - sk / 2 + centroid
            d_tl = -d_Yla / 2 - dsk / 2 + d_centroid
            d_Iz = d_Iz + np.sum(
                d_Xld * sk * tl**2 + Xld * dsk * tl**2 + Xld * sk * 2 * tl * d_tl, axis=0
            )
            # spars
            cf = cf0 - centroid
            d_cf = d_cf0 - d_centroid
            d_Iz = d_Iz + (
                1.0 / 12.0 * (dst * hfs**3 + st * 3 * hfs**2 * d_hfs)
                + dst * hfs * cf**2
                + st * d_hfs * cf**2
                + st * hfs * 2 * cf * d_cf
            )
            cr = cr0 - centroid
            d_cr = d_cr0 - d_centroid
            d_Iz = d_Iz + (
                1.0 / 12.0 * (dst * hrs**3 + st * 3 * hrs**2 * d_hrs)
                + dst * hrs * cr**2
                + st * d_hrs * cr**2
                + st * hrs * 2 * cr * d_cr
            )

            # ----- Qz -----
            tq = (Yua / 2 - sk / 2) - centroid
            d_tq = d_Yua / 2 - dsk / 2 - d_centroid
            d_Qz = np.sum(d_tq * sk * Xud + tq * dsk * Xud + tq * sk * d_Xud, axis=0)
            qf = RYu[0] - sk - centroid
            d_qf = d_RYu[0] - dsk - d_centroid
            d_Qz = d_Qz + qf * d_qf * st + qf**2 / 2 * dst
            qr = RYu[-1] - sk - centroid
            d_qr = d_RYu[-1] - dsk - d_centroid
            d_Qz = d_Qz + qr * d_qr * st + qr**2 / 2 * dst

            # ----- Iy (backward bending) -----
            sfront = RYu[0] - RYl[0]
            d_sfront = d_RYu[0] - d_RYl[0]
            srear = RYu[-1] - RYl[-1]
            d_srear = d_RYu[-1] - d_RYl[-1]
            fmaf = sfront * st * (RXu[0] + st / 2)
            d_fmaf = d_sfront * st * (RXu[0] + st / 2) + sfront * dst * (RXu[0] + st / 2) + sfront * st * (
                d_RXu[0] + dst / 2
            )
            fmar = srear * st * (RXu[-1] - st / 2)
            d_fmar = d_srear * st * (RXu[-1] - st / 2) + srear * dst * (RXu[-1] - st / 2) + srear * st * (
                d_RXu[-1] - dst / 2
            )
            denomIv = (sfront + srear) * st
            d_denomIv = (d_sfront + d_srear) * st + (sfront + srear) * dst
            cIv = (fmaf + fmar) / denomIv
            d_cIv = ((d_fmaf + d_fmar) * denomIv - (fmaf + fmar) * d_denomIv) / denomIv**2

            ef = cIv - (RXu[0] + st / 2)
            d_ef = d_cIv - (d_RXu[0] + dst / 2)
            d_Iy = 1.0 / 12.0 * (d_sfront * st**3 + sfront * 3 * st**2 * dst) + (
                d_sfront * st + sfront * dst
            ) * ef**2 + sfront * st * 2 * ef * d_ef
            er = RXu[-1] - st / 2 - cIv
            d_er = d_RXu[-1] - dst / 2 - d_cIv
            d_Iy = d_Iy + 1.0 / 12.0 * (d_srear * st**3 + srear * 3 * st**2 * dst) + (
                d_srear * st + srear * dst
            ) * er**2 + srear * st * 2 * er * d_er
            w = RXu[-1] - RXu[0] - 2 * st
            d_w = d_RXu[-1] - d_RXu[0] - 2 * dst
            cc = cIv - (RXu[-1] + RXu[0]) / 2
            d_cc = d_cIv - (d_RXu[-1] + d_RXu[0]) / 2
            d_Iy = d_Iy + 2.0 * (
                1.0 / 12.0 * (dsk * w**3 + sk * 3 * w**2 * d_w)
                + dsk * w * cc**2
                + sk * d_w * cc**2
                + sk * w * 2 * cc * d_cc
            )

            # ----- htop, hbottom (KS aggregation; max terms cancel in the derivative) -----
            mmax = np.max(RYu, axis=0, keepdims=True)
            eu = np.exp(ks_rho * (RYu - mmax))
            wu = eu / np.sum(eu, axis=0, keepdims=True)
            d_htop = np.sum(wu * d_RYu, axis=0) - d_centroid

            mmaxl = np.max(-RYl, axis=0, keepdims=True)
            el = np.exp(ks_rho * (-RYl - mmaxl))
            wl = el / np.sum(el, axis=0, keepdims=True)
            d_hbottom = np.sum(wl * (-d_RYl), axis=0) + d_centroid

            # ----- hfront, hrear -----
            d_hfront = d_cIv - d_RXu[0]
            d_hrear = d_RXu[-1] - d_cIv

            return {
                "A": d_area,
                "A_enc": d_Aenc,
                "A_int": d_Aint,
                "Iy": d_Iy,
                "Qz": d_Qz,
                "Iz": d_Iz,
                "J": d_J,
                "htop": d_htop,
                "hbottom": d_hbottom,
                "hfront": d_hfront,
                "hrear": d_hrear,
            }

        nseg = self.ny - 1
        one = np.ones(nseg)
        zero = np.zeros(nseg)
        seeds = {
            "fem_chords": (one, zero, zero, zero, zero, zero),
            "spar_thickness": (zero, one, zero, zero, zero, zero),
            "skin_thickness": (zero, zero, one, zero, zero, zero),
            "t_over_c": (zero, zero, zero, one, zero, zero),
            "streamwise_chords": (zero, zero, zero, zero, one, zero),
            "fem_twists": (zero, zero, zero, zero, zero, one),
        }
        for inp, seed in seeds.items():
            derivs = fwd(*seed)
            for out, dval in derivs.items():
                partials[out, inp] = dval
