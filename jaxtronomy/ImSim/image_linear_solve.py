from jaxtronomy.ImSim.image_model import ImageModel
import lenstronomy.ImSim.de_lens as de_lens
from jaxtronomy.Util import util
from jaxtronomy.ImSim.Numerics.convolution import PixelKernelConvolution
import numpy as np

from jax import jit, numpy as jnp
from functools import partial

__all__ = ["ImageLinearFit"]


class ImageLinearFit(ImageModel):
    """Linear version class, inherits ImageModel.

    When light models use pixel-based profile types, such as 'SLIT_STARLETS', the WLS
    linear inversion is replaced by the regularized inversion performed by an external
    solver. The current pixel-based solver is provided by the SLITronomy plug-in.
    """

    def __init__(
        self,
        data_class,
        psf_class=None,
        lens_model_class=None,
        source_model_class=None,
        lens_light_model_class=None,
        point_source_class=None,
        extinction_class=None,
        kwargs_numerics=None,
        likelihood_mask=None,
        psf_error_map_bool_list=None,
        kwargs_pixelbased=None,
        linear_solver=False,
    ):
        """

        :param data_class: ImageData() instance
        :param psf_class: PSF() instance
        :param lens_model_class: LensModel() instance
        :param source_model_class: LightModel() instance
        :param lens_light_model_class: LightModel() instance
        :param point_source_class: PointSource() instance
        :param kwargs_numerics: keyword arguments passed to the Numerics module
        :param likelihood_mask: 2d boolean array of pixels to be counted in the likelihood calculation
        :param psf_error_map_bool_list: list of boolean of length of point source models.
         Indicates whether PSF error map is used for the point source model stated as the index.
        :param kwargs_pixelbased: keyword arguments with various settings related to the pixel-based solver
         (see SLITronomy documentation) being applied to the point sources. not supported in jaxtronomy
        :param linear_solver: bool, not supported in jaxtronomy
        """
        super(ImageLinearFit, self).__init__(
            data_class,
            psf_class=psf_class,
            lens_model_class=lens_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
            point_source_class=point_source_class,
            extinction_class=extinction_class,
            kwargs_numerics=kwargs_numerics,
            kwargs_pixelbased=kwargs_pixelbased,
        )
        if linear_solver:
            raise ValueError("linear solve is not supported in jaxtronomy") 
        self._linear_solver = linear_solver
        if psf_error_map_bool_list is None:
            psf_error_map_bool_list = [True] * len(
                self.PointSource.point_source_type_list
            )
        self._psf_error_map_bool_list = psf_error_map_bool_list
        if likelihood_mask is None:
            likelihood_mask = jnp.ones_like(data_class.data)
        self.likelihood_mask = jnp.array(likelihood_mask, dtype=bool)
        self._mask1d = util.image2array(self.likelihood_mask)
        #if self._pixelbased_bool is True:
        #    # update the pixel-based solver with the likelihood mask
        #    self.PixelSolver.set_likelihood_mask(self.likelihood_mask)

        # prepare to use fft convolution for the natwt linear solver
        if self.Data.likelihood_method() == "interferometry_natwt":
            self._convolution = PixelKernelConvolution(
                kernel=self.PSF.kernel_point_source
            )

        # Compile class functions
        self.reduced_residuals = jit(self._reduced_residuals)
        self.likelihood_data_given_model = jit(self._likelihood_data_given_model, static_argnames=("check_positive_flux"))
        self.array_masked2image = jit(self._array_masked2image)
        self.error_response = jit(self._error_response)
        self._error_map_psf = jit(self.__error_map_psf)
        self.reduced_chi2 = jit(self._reduced_chi2)
    
    def image_pixelbased_solve(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        init_lens_light_model=None,
    ):
        raise Exception("image_pixelbased_solve is not supported in jaxtronomy")    

    @property
    def data_response(self):
        """Returns the 1d array of the data element that is fitted for (including
        masking)
        :return: 1d numpy array
        """
        d = self.image2array_masked(self.Data.data)
        return d

    # Don't apply JIT decorator to this function
    def _error_response(self, kwargs_lens, kwargs_ps, kwargs_special):
        """Returns the 1d array of the error estimate corresponding to the data
        response.

        :return: 1d numpy array of response, 2d array of additional errors (e.g. point
            source uncertainties)
        """
        model_error = self._error_map_model(
            kwargs_lens, kwargs_ps, kwargs_special=kwargs_special
        )
        # adding the uncertainties estimated from the data with the ones from the model
        C_D_response = self.image2array_masked(self.Data.C_D + model_error)
        return C_D_response, model_error

    # Don't apply JIT decorator to this function
    def _likelihood_data_given_model(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        source_marg=False,
        linear_prior=None,
        check_positive_flux=False,
        linear_solver=False,
    ):
        """Computes the likelihood of the data given a model This is specified with the
        non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens: list of keyword arguments corresponding to the superposition
            of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the
            superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different
            lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as
            external shear and point source image positions
        :param source_marg: bool, performs a marginalization over the linear parameters
        :param linear_prior: linear prior width in eigenvalues
        :param check_positive_flux: bool, if True, checks whether the linear inversion
            resulted in non-negative flux components and applies a punishment in the
            likelihood if so.
        :param linear_solver: bool, if True (default) fixes the linear amplitude
            parameters 'amp' (avoid sampling) such that they get overwritten by the
            linear solver solution.
        :return: log likelihood (natural logarithm), linear parameter list
        """
        # generate image
        if linear_solver is False:
            im_sim = self.image(
                kwargs_lens,
                kwargs_source,
                kwargs_lens_light,
                kwargs_ps,
                kwargs_extinction,
                kwargs_special,
            )
            cov_matrix, param = None, None
            model_error = self._error_map_model(
                kwargs_lens, kwargs_ps=kwargs_ps, kwargs_special=kwargs_special
            )
        else:
            raise ValueError("Linear solver not supported in JAXtronomy. Use lenstronomy instead")
        # compute X^2
        logL = self.likelihood_data_given_model_solution(
            im_sim,
            model_error,
            cov_matrix,
            param,
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
            source_marg=source_marg,
            linear_prior=linear_prior,
            check_positive_flux=check_positive_flux,
        )
        return logL, param

    # Don't apply JIT decorator to this function
    def likelihood_data_given_model_solution(
        self,
        model,
        model_error,
        cov_matrix,
        param,
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light,
        kwargs_ps,
        source_marg=False,
        linear_prior=None,
        check_positive_flux=False,
    ):
        """

        :param model:
        :param model_error:
        :param cov_matrix:
        :param param:
        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param source_marg:
        :param linear_prior:
        :param check_positive_flux:
        :return:
        """

        logL = self.Data.log_likelihood(model, self.likelihood_mask, model_error)

        #if self._pixelbased_bool is False:
        #    if cov_matrix is not None and source_marg:
        #        marg_const = de_lens.marginalization_new(
        #            cov_matrix, d_prior=linear_prior
        #        )
        #        logL += marg_const
        if check_positive_flux is True:
            raise ValueError("check positive flux is not supported in jaxtronomy due to issues with autodifferentiation")
            #bool_ = self.check_positive_flux(
            #    kwargs_source, kwargs_lens_light, kwargs_ps
            #)
            #if bool_ is False:
            #    logL -= 10**8
        return logL
    
    #def update_pixel_kwargs(self, kwargs_source, kwargs_lens_light):
        """Update kwargs arguments for pixel-based profiles with fixed properties such
        as their number of pixels, scale, and center coordinates (fixed to the origin).

        :param kwargs_source: list of keyword arguments corresponding to the
            superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to the
            superposition of different lens light profiles
        :return: updated kwargs_source and kwargs_lens_light
        """
        # in case the source plane grid size has changed, update the kwargs accordingly
        ss_factor_source = self.SourceNumerics.grid_supersampling_factor
        kwargs_source[0]["n_pixels"] = int(
            self.Data.num_pixel * ss_factor_source**2
        )  #  effective number of pixels in source plane
        kwargs_source[0]["scale"] = (
            self.Data.pixel_width / ss_factor_source
        )  # effective pixel size of source plane grid
        # pixelated reconstructions have no well-defined center, we put it arbitrarily at (0, 0), center of the image
        kwargs_source[0]["center_x"] = 0
        kwargs_source[0]["center_y"] = 0
        # do the same if the lens light has been reconstructed
        if kwargs_lens_light is not None and len(kwargs_lens_light) > 0:
            kwargs_lens_light[0]["n_pixels"] = self.Data.num_pixel
            kwargs_lens_light[0]["scale"] = self.Data.pixel_width
            kwargs_lens_light[0]["center_x"] = 0
            kwargs_lens_light[0]["center_y"] = 0
        return kwargs_source, kwargs_lens_light

    # Don't apply JIT decorator to this function
    def _reduced_residuals(self, model, error_map=0):
        """

        :param model: 2d numpy array of the modeled image
        :param error_map: 2d numpy array of additional noise/error terms from model components (such as PSF model uncertainties)
        :return: 2d numpy array of reduced residuals per pixel
        """
        mask = self.likelihood_mask
        C_D = self.Data.C_D_model(model)
        residual = (model - self.Data.data) / jnp.sqrt(C_D + jnp.abs(error_map)) * mask
        return residual

    # Don't apply JIT decorator to this function
    def _reduced_chi2(self, model, error_map=0):
        """Returns reduced chi2 :param model: 2d numpy array of a model predicted image
        :param error_map: same format as model, additional error component (such as PSF
        errors) :return: reduced chi2."""
        norm_res = self.reduced_residuals(model, error_map)
        return jnp.sum(norm_res**2) / self.num_data_evaluate

    @property
    def num_data_evaluate(self):
        """Number of pixels included in the likelihood calculation

        :return: number of evaluated pixels
        :rtype: int.
        """
        return jnp.sum(self.likelihood_mask).astype(int)

    def update_data(self, data_class):
        """

        :param data_class: instance of Data() class
        :return: no return. Class is updated.
        """
        self.Data = data_class
        self.ImageNumerics._PixelGrid = data_class

        # Recompile all functions involving self.Data
        self.reduced_residuals = jit(self._reduced_residuals)
        self.reduced_chi2 = jit(self._reduced_chi2)
        self.array_masked2image = jit(self._array_masked2image)
        self.error_response = jit(self._error_response)
        self._error_map_psf = jit(self.__error_map_psf)
        self.likelihood_data_given_model = jit(self._likelihood_data_given_model, static_argnames=("check_positive_flux"))

    @partial(jit, static_argnums=0)
    def image2array_masked(self, image):
        """Returns 1d array of values in image that are not masked out for the
        likelihood computation/linear minimization :param image: 2d numpy array of full
        image :return: 1d array."""
        array = util.image2array(image)
        return array[self._mask1d]

    # Don't apply JIT decorator to this function
    def _array_masked2image(self, array):
        """

        :param array: 1d array of values not masked out
        :return: 2d array of full image
        """
        nx, ny = self.Data.num_pixel_axes
        grid1d = jnp.zeros(nx * ny)
        grid1d = grid1d.at[self._mask1d].set(array)
        grid2d = util.array2image(grid1d, nx, ny)
        return grid2d

    # Don't apply JIT decorator to this function
    def _error_map_model(self, kwargs_lens, kwargs_ps, kwargs_special=None):
        """Noise estimate (variances as diagonal of the pixel covariance matrix)
        resulted from inherent model uncertainties This term is currently the psf error
        map.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_ps: point source keyword arguments
        :param kwargs_special: special parameter keyword arguments
        :return: 2d array corresponding to the pixels in terms of variance in noise
        """
        return self._error_map_psf(kwargs_lens, kwargs_ps, kwargs_special)

    # Don't apply JIT decorator to this function
    def __error_map_psf(self, kwargs_lens, kwargs_ps, kwargs_special=None):
        """Map of image with error terms (sigma**2) expected from inaccuracies in the
        PSF modeling.

        :param kwargs_lens: lens model keyword arguments
        :param kwargs_ps: point source keyword arguments
        :param kwargs_special: special parameter keyword arguments
        :return: 2d array of size of the image
        """
        error_map = jnp.zeros(self.Data.num_pixel_axes)
        if self._psf_error_map is True:
            for k, bool_ in enumerate(self._psf_error_map_bool_list):
                if bool_ is True:
                    ra_pos, dec_pos, _ = self.PointSource.point_source_list(
                        kwargs_ps, kwargs_lens=kwargs_lens, k=k, with_amp=False
                    )
                    if len(ra_pos) > 0:
                        # ra_pos, dec_pos = self._displace_astrometry(
                        #     ra_pos, dec_pos, kwargs_special=kwargs_special
                        # )
                        error_map += self.ImageNumerics.psf_error_map(
                            ra_pos,
                            dec_pos,
                            None,
                            self.Data.data,
                            fix_psf_error_map=False,
                        )
        return error_map

    #def check_positive_flux(self, kwargs_source, kwargs_lens_light, kwargs_ps):
        """Checks whether the surface brightness profiles contain positive fluxes and
        returns bool if True.

        :param kwargs_source: source surface brightness keyword argument list
        :param kwargs_lens_light: lens surface brightness keyword argument list
        :param kwargs_ps: point source keyword argument list
        :return: boolean
        """
        pos_bool_ps = self.PointSource.check_positive_flux(kwargs_ps)
        if self._pixelbased_bool is True:
            # this constraint must be handled by the pixel-based solver
            pos_bool_source = True
            pos_bool_lens_light = True
        else:
            pos_bool_source = self.SourceModel.check_positive_flux_profile(
                kwargs_source
            )
            pos_bool_lens_light = self.LensLightModel.check_positive_flux_profile(
                kwargs_lens_light
            )
        if (
            pos_bool_ps is True
            and pos_bool_source is True
            and pos_bool_lens_light is True
        ):
            return True
        else:
            return False