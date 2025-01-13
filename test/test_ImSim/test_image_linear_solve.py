__author__ = "sibirrer"

import numpy.testing as npt
import numpy as np
import pytest

import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.psf import PSF

from lenstronomy.Data.imaging_data import ImageData as ImageData_ref
from lenstronomy.LensModel.lens_model import LensModel as LensModel_ref
from lenstronomy.LightModel.light_model import LightModel as LightModel_ref
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit as ImageLinearFit_ref

from jaxtronomy.Data.imaging_data import ImageData
from jaxtronomy.LensModel.lens_model import LensModel
from jaxtronomy.LightModel.light_model import LightModel
from jaxtronomy.ImSim.image_linear_solve import ImageLinearFit


class TestImageLinearFit(object):
    """Tests the source model routines."""

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg, inverse=True
        )

        # Create likelihood mask
        likelihood_mask = np.ones((numPix, numPix))
        likelihood_mask[50][::2] -= likelihood_mask[50][::2]
        likelihood_mask[25][::3] -= likelihood_mask[25][::3]
        self.likelihood_mask = likelihood_mask

        self.data_class = ImageData(**kwargs_data)
        self.data_class_ref = ImageData_ref(**kwargs_data)

        # make a second data class for update_data testing
        sigma_bkg = 0.08  # background noise per pixel
        exp_time = 137  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)

        # PSF specification
        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg, inverse=True
        )

        self.data_class2 = ImageData(**kwargs_data)
        self.data_class2_ref = ImageData_ref(**kwargs_data)        

        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        self.psf_class_gaussian = PSF(**kwargs_psf)
        kernel = self.psf_class_gaussian.kernel_point_source
        kwargs_psf = {
            "psf_type": "PIXEL",
            "kernel_point_source": kernel,
            "psf_error_map": np.ones_like(kernel) * 0.001 * kernel**2,
        }
        self.psf_class = PSF(**kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {
            "gamma1": 0.01,
            "gamma2": 0.01,
        }  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {
            "theta_E": 1.0,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_model_list = ["SIE", "SHEAR"]
        lens_model_class_ref = LensModel_ref(lens_model_list=lens_model_list)
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {
            "amp": 31.0,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {
            "amp": 23.0,
            "R_sersic": 0.6,
            "n_sersic": 7,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        lens_light_model_class_ref = LightModel_ref(light_model_list=lens_light_model_list)

        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        source_model_class_ref = LightModel_ref(light_model_list=source_model_list)

        kwargs_numerics = {
            "supersampling_factor": 3,
            "supersampling_convolution": True,
        }
        self.imagelinearfit = ImageLinearFit(
            self.data_class,
            self.psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            kwargs_numerics=kwargs_numerics,
            likelihood_mask=likelihood_mask,
        )
        self.imagelinearfit_ref = ImageLinearFit_ref(
            self.data_class_ref,
            self.psf_class,
            lens_model_class_ref,
            source_model_class_ref,
            lens_light_model_class_ref,
            kwargs_numerics=kwargs_numerics,
            likelihood_mask=likelihood_mask,
        )
        image_sim = sim_util.simulate_simple(
            self.imagelinearfit_ref,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            no_noise=True
        )
        self.data_class.update_data(image_sim + 1.03)
        self.data_class_ref.update_data(image_sim + 1.03)
        self.data_class2.update_data(image_sim - 1.02)
        self.data_class2_ref.update_data(image_sim - 1.02)

    def test_data_response(self):
        npt.assert_array_almost_equal(self.imagelinearfit.data_response, self.imagelinearfit_ref.data_response)

    def test_error(self):
        cd_response, model_error = self.imagelinearfit.error_response(self.kwargs_lens, [], [])
        cd_response_ref, model_error_ref = self.imagelinearfit_ref.error_response(self.kwargs_lens, [], [])
        npt.assert_array_almost_equal(cd_response, cd_response_ref, decimal=8)
        npt.assert_array_almost_equal(model_error, model_error_ref, decimal=8)

        error = self.imagelinearfit._error_map_psf(self.kwargs_lens, [], [])
        error_ref = self.imagelinearfit_ref._error_map_psf(self.kwargs_lens, [], [])
        npt.assert_array_almost_equal(error, error_ref, decimal=8)

        self.imagelinearfit.update_data(self.data_class2)
        self.imagelinearfit_ref.update_data(self.data_class2_ref)

        cd_response, model_error = self.imagelinearfit.error_response(self.kwargs_lens, [], [])
        cd_response_ref, model_error_ref = self.imagelinearfit_ref.error_response(self.kwargs_lens, [], [])
        npt.assert_array_almost_equal(cd_response, cd_response_ref, decimal=8)
        npt.assert_array_almost_equal(model_error, model_error_ref, decimal=8)

        error = self.imagelinearfit._error_map_psf(self.kwargs_lens, [], [])
        error_ref = self.imagelinearfit_ref._error_map_psf(self.kwargs_lens, [], [])
        npt.assert_array_almost_equal(error, error_ref, decimal=8)

    def test_likelihood_data_given_model(self):
        likelihood, _ = self.imagelinearfit.likelihood_data_given_model(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        likelihood_ref, _ = self.imagelinearfit_ref.likelihood_data_given_model(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            linear_solver=False
        )
        npt.assert_array_almost_equal(likelihood, likelihood_ref, decimal=8)

        self.imagelinearfit.update_data(self.data_class2)
        self.imagelinearfit_ref.update_data(self.data_class2_ref)

        likelihood, _ = self.imagelinearfit.likelihood_data_given_model(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        likelihood_ref, _ = self.imagelinearfit_ref.likelihood_data_given_model(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            linear_solver=False
        )
        npt.assert_array_almost_equal(likelihood, likelihood_ref, decimal=8)

    def test_reduced_residuals(self):
        model = self.data_class.data
        residuals = self.imagelinearfit.reduced_residuals(model)
        residuals_ref = self.imagelinearfit_ref.reduced_residuals(model)
        npt.assert_array_almost_equal(residuals, residuals_ref, decimal=8)

        error_map = np.ones_like(model) * 0.5
        residuals = self.imagelinearfit.reduced_residuals(model, error_map)
        residuals_ref = self.imagelinearfit_ref.reduced_residuals(model, error_map)
        npt.assert_array_almost_equal(residuals, residuals_ref, decimal=8)

        self.imagelinearfit.update_data(self.data_class2)
        self.imagelinearfit_ref.update_data(self.data_class2_ref)

        residuals = self.imagelinearfit.reduced_residuals(model)
        residuals_ref = self.imagelinearfit_ref.reduced_residuals(model)
        npt.assert_array_almost_equal(residuals, residuals_ref, decimal=8)

        residuals = self.imagelinearfit.reduced_residuals(model, error_map)
        residuals_ref = self.imagelinearfit_ref.reduced_residuals(model, error_map)
        npt.assert_array_almost_equal(residuals, residuals_ref, decimal=8)

    def test_reduced_chi2(self):
        model = self.data_class.data
        chi2 = self.imagelinearfit.reduced_chi2(model)
        chi2_ref = self.imagelinearfit_ref.reduced_chi2(model)
        npt.assert_array_almost_equal(chi2, chi2_ref, decimal=8)

        error_map = np.ones_like(model) * 0.5
        chi2 = self.imagelinearfit.reduced_chi2(model, error_map)
        chi2_ref = self.imagelinearfit_ref.reduced_chi2(model, error_map)
        npt.assert_array_almost_equal(chi2, chi2_ref, decimal=8)

        self.imagelinearfit.update_data(self.data_class2)
        self.imagelinearfit_ref.update_data(self.data_class2_ref)

        chi2 = self.imagelinearfit.reduced_chi2(model)
        chi2_ref = self.imagelinearfit_ref.reduced_chi2(model)
        npt.assert_array_almost_equal(chi2, chi2_ref, decimal=8)

        chi2 = self.imagelinearfit.reduced_chi2(model, error_map)
        chi2_ref = self.imagelinearfit_ref.reduced_chi2(model, error_map)
        npt.assert_array_almost_equal(chi2, chi2_ref, decimal=8)


    def test_image2array_masked(self):
        image = self.data_class.data
        array = self.imagelinearfit.image2array_masked(image)
        array_ref = self.imagelinearfit_ref.image2array_masked(image)
        npt.assert_array_almost_equal(array, array_ref, decimal=8)

    def test_array_masked2image(self):
        image_0 = self.data_class.data
        array = self.imagelinearfit.image2array_masked(image_0)
        array_ref = self.imagelinearfit_ref.image2array_masked(image_0)

        image = self.imagelinearfit.array_masked2image(array)
        image_ref = self.imagelinearfit_ref.array_masked2image(array_ref)
        npt.assert_array_almost_equal(image, image_ref, decimal=8)
        npt.assert_array_almost_equal(image, image_0 * self.likelihood_mask, decimal=8)

    def test_raises(self):
        npt.assert_raises(ValueError, ImageLinearFit, self.data_class, self.psf_class, linear_solver=True)
        npt.assert_raises(Exception, self.imagelinearfit.image_pixelbased_solve)
        npt.assert_raises(ValueError, self.imagelinearfit.likelihood_data_given_model,
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            check_positive_flux=True
        )
        npt.assert_raises(ValueError, self.imagelinearfit.likelihood_data_given_model,
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            linear_solver=True
        )


if __name__ == "__main__":
    pytest.main()