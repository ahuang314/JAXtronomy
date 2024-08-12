__author__ = "sibirrer"

from lenstronomy.LensModel.Profiles.epl import EPL as EPL_ref
from lenstronomy.LensModel.Profiles.epl import EPLMajorAxis as EPLMajorAxis_ref
from lenstronomy.LensModel.Profiles.epl import EPLQPhi as EPLQPhi_ref
from jaxtronomy.LensModel.Profiles.epl import EPL, EPLMajorAxis, EPLQPhi

import numpy as np
import numpy.testing as npt
import pytest
import jax

jax.config.update("jax_enable_x64", True)  # 64-bit floats, consistent with numpy
import jax.numpy as jnp


class TestEPL(object):

    def setup_method(self):
        self.profile = EPL()
        self.profile_ref = EPL_ref()

    def test_param_conv(self):
        theta_E = 12.3
        gamma, e1, e2 = 1.7, 0.3, -0.4
        self.profile.set_static(theta_E, gamma, e1, e2)
        self.profile_ref.set_static(theta_E, gamma, e1, e2)

        b_static, t_static, q_static, phi_static = self.profile.param_conv(theta_E, gamma, e1, e2)

        (
            b_static_ref,
            t_static_ref,
            q_static_ref,
            phi_static_ref,
        ) = self.profile_ref.param_conv(theta_E, gamma, e1, e2)
        npt.assert_almost_equal(b_static, b_static_ref, decimal=8)
        npt.assert_almost_equal(t_static, t_static_ref, decimal=8)
        npt.assert_almost_equal(q_static, q_static_ref, decimal=8)
        npt.assert_almost_equal(phi_static, phi_static_ref, decimal=8)

        theta_E = 11.3
        gamma, e1, e2 = 1.6, 0.2, -0.3
        self.profile.set_dynamic()
        self.profile_ref.set_dynamic()
        b, t, q, phi_G = self.profile.param_conv(theta_E, gamma, e1, e2)
        b_ref, t_ref, q_ref, phi_G_ref = self.profile_ref.param_conv(
            theta_E, gamma, e1, e2
        )
        npt.assert_almost_equal(b, b_ref, decimal=8)
        npt.assert_almost_equal(t, t_ref, decimal=8)
        npt.assert_almost_equal(q, q_ref, decimal=8)
        npt.assert_almost_equal(phi_G, phi_G_ref, decimal=8)

        # Do the same test as before to make sure that the
        # static variables are correctly updated with new values
        theta_E = 4.3
        gamma, e1, e2 = 2.3, 0.4, -0.2
        self.profile.set_static(theta_E, gamma, e1, e2)
        self.profile_ref.set_static(theta_E, gamma, e1, e2)

        b_static, t_static, q_static, phi_static = self.profile.param_conv(theta_E, gamma, e1, e2)

        (
            b_static_ref,
            t_static_ref,
            q_static_ref,
            phi_static_ref,
        ) = self.profile_ref.param_conv(theta_E, gamma, e1, e2)
        npt.assert_almost_equal(b_static, b_static_ref, decimal=8)
        npt.assert_almost_equal(t_static, t_static_ref, decimal=8)
        npt.assert_almost_equal(q_static, q_static_ref, decimal=8)
        npt.assert_almost_equal(phi_static, phi_static_ref, decimal=8)

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 12.3
        gamma, e1, e2 = 1.7, 0.3, -0.2
        values = self.profile.function(x, y, theta_E, gamma, e1, e2)
        values_ref = self.profile_ref.function(x, y, theta_E, gamma, e1, e2)
        npt.assert_almost_equal(values, values_ref, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(x, y, theta_E, gamma, e1, e2)
        values_ref = self.profile_ref.function(x, y, theta_E, gamma, e1, e2)
        npt.assert_almost_equal(values, values_ref, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        gamma, e1, e2 = 1.7, -0.1, 0.2
        f_x, f_y = self.profile.derivatives(x, y, theta_E, gamma, e1, e2)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, theta_E, gamma, e1, e2)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = self.profile.derivatives(x, y, theta_E, gamma, e1, e2)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, theta_E, gamma, e1, e2)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        gamma, e1, e2 = 2.2, 0.1, -0.4
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, theta_E, gamma, e1, e2)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, theta_E, gamma, e1, e2
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, theta_E, gamma, e1, e2)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, theta_E, gamma, e1, e2
        )
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_yx, decimal=8)

    def test_mass_3d_lens(self):
        mass = self.profile.mass_3d_lens(r=1, theta_E=1, gamma=1.7)
        mass_ref = self.profile_ref.mass_3d_lens(r=1, theta_E=1, gamma=1.7)
        npt.assert_almost_equal(mass, mass_ref, decimal=8)

    def test_density_lens(self):
        x = 1
        y = 2
        theta_E = 1.0
        r = jnp.sqrt(x**2 + y**2)
        gamma = 2.1
        density = self.profile.density_lens(r, theta_E, gamma)
        density_ref = self.profile_ref.density_lens(r, theta_E, gamma)
        npt.assert_almost_equal(density, density_ref, decimal=8)

    def test_jax_jit(self):
        x = jnp.array([1])
        y = jnp.array([2])
        theta_E = 12.3
        gamma, e1, e2 = 1.7, 0.1, -0.3
        jitted = jax.jit(self.profile.function)
        npt.assert_almost_equal(
            self.profile.function(x, y, theta_E, gamma, e1, e2),
            jitted(x, y, theta_E, gamma, e1, e2),
            decimal=8,
        )

        jitted = jax.jit(self.profile.derivatives)
        npt.assert_array_almost_equal(
            self.profile.derivatives(x, y, theta_E, gamma, e1, e2),
            jitted(x, y, theta_E, gamma, e1, e2),
            decimal=8,
        )

        jitted = jax.jit(self.profile.hessian)
        npt.assert_array_almost_equal(
            self.profile.hessian(x, y, theta_E, gamma, e1, e2),
            jitted(x, y, theta_E, gamma, e1, e2),
            decimal=8,
        )


class TestEPLMajorAxis(object):

    def setup_method(self):
        self.profile = EPLMajorAxis()
        self.profile_ref = EPLMajorAxis_ref()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 12.3
        b, t, q = 1.7, 0.7, 0.7
        values = self.profile.function(x, y, b, t, q)
        values_ref = self.profile_ref.function(x, y, b, t, q)
        npt.assert_almost_equal(values, values_ref, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(x, y, b, t, q)
        values_ref = self.profile_ref.function(x, y, b, t, q)
        npt.assert_almost_equal(values, values_ref, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        b, t, q = 1.7, 0.9, 0.8
        f_x, f_y = self.profile.derivatives(x, y, b, t, q)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, b, t, q)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = self.profile.derivatives(x, y, b, t, q)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, b, t, q)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        b, t, q = 2.2, 0.6, 0.4
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, b, t, q)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(x, y, b, t, q)
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=7)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=7)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=7)
        npt.assert_almost_equal(f_xy, f_yx, decimal=7)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, b, t, q)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(x, y, b, t, q)
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=7)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=7)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=7)
        npt.assert_array_almost_equal(f_xy, f_yx, decimal=7)

    def test_jax_jit(self):
        x = jnp.array([1])
        y = jnp.array([2])
        b, t, q = 3.7, 1.1, 0.3
        jitted = jax.jit(self.profile.function)
        npt.assert_almost_equal(
            self.profile.function(x, y, b, t, q),
            jitted(x, y, b, t, q),
            decimal=8,
        )

        jitted = jax.jit(self.profile.derivatives)
        npt.assert_array_almost_equal(
            self.profile.derivatives(x, y, b, t, q),
            jitted(x, y, b, t, q),
            decimal=8,
        )

        jitted = jax.jit(self.profile.hessian)
        npt.assert_array_almost_equal(
            self.profile.hessian(x, y, b, t, q),
            jitted(x, y, b, t, q),
            decimal=8,
        )


class TestEPLQPhi(object):

    def setup_method(self):
        self.profile = EPLQPhi()
        self.profile_ref = EPLQPhi_ref()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 12.3
        gamma, q, phi = 1.7, 0.3, -2.2
        values = self.profile.function(x, y, theta_E, gamma, q, phi)
        values_ref = self.profile_ref.function(x, y, theta_E, gamma, q, phi)
        npt.assert_almost_equal(values, values_ref, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(x, y, theta_E, gamma, q, phi)
        values_ref = self.profile_ref.function(x, y, theta_E, gamma, q, phi)
        npt.assert_almost_equal(values, values_ref, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        gamma, q, phi = 1.7, 0.7, 2.2
        f_x, f_y = self.profile.derivatives(x, y, theta_E, gamma, q, phi)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, theta_E, gamma, q, phi)
        npt.assert_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_almost_equal(f_y, f_y_ref, decimal=8)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_x, f_y = self.profile.derivatives(x, y, theta_E, gamma, q, phi)
        f_x_ref, f_y_ref = self.profile_ref.derivatives(x, y, theta_E, gamma, q, phi)
        npt.assert_array_almost_equal(f_x, f_x_ref, decimal=8)
        npt.assert_array_almost_equal(f_y, f_y_ref, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        gamma, q, phi = 2.2, 0.4, 1.4
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, theta_E, gamma, q, phi)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, theta_E, gamma, q, phi
        )
        npt.assert_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_ref, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, theta_E, gamma, q, phi)
        f_xx_ref, f_xy_ref, f_yx_ref, f_yy_ref = self.profile_ref.hessian(
            x, y, theta_E, gamma, q, phi
        )
        npt.assert_array_almost_equal(f_xx, f_xx_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_xy_ref, decimal=8)
        npt.assert_array_almost_equal(f_yy, f_yy_ref, decimal=8)
        npt.assert_array_almost_equal(f_xy, f_yx, decimal=8)

    def test_mass_3d_lens(self):
        mass = self.profile.mass_3d_lens(r=1, theta_E=1, gamma=1.7)
        mass_ref = self.profile_ref.mass_3d_lens(r=1, theta_E=1, gamma=1.7)
        npt.assert_almost_equal(mass, mass_ref, decimal=8)

    def test_density_lens(self):
        x = 3
        y = 2
        theta_E = 1.0
        r = jnp.sqrt(x**2 + y**2)
        gamma = 2.1
        density = self.profile.density_lens(r, theta_E, gamma)
        density_ref = self.profile_ref.density_lens(r, theta_E, gamma)
        npt.assert_almost_equal(density, density_ref, decimal=8)

    def test_jax_jit(self):
        x = jnp.array([1])
        y = jnp.array([2])
        theta_E = 12.3
        gamma, q, phi = 1.7, 0.3, 3.1
        jitted = jax.jit(self.profile.function)
        npt.assert_almost_equal(
            self.profile.function(x, y, theta_E, gamma, q, phi),
            jitted(x, y, theta_E, gamma, q, phi),
            decimal=8,
        )

        jitted = jax.jit(self.profile.derivatives)
        npt.assert_array_almost_equal(
            self.profile.derivatives(x, y, theta_E, gamma, q, phi),
            jitted(x, y, theta_E, gamma, q, phi),
            decimal=8,
        )

        jitted = jax.jit(self.profile.hessian)
        npt.assert_array_almost_equal(
            self.profile.hessian(x, y, theta_E, gamma, q, phi),
            jitted(x, y, theta_E, gamma, q, phi),
            decimal=8,
        )


if __name__ == "__main__":
    pytest.main()
