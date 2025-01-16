from jaxtronomy.ImSim.image_linear_solve import ImageLinearFit
from jaxtronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from jaxtronomy.Util import class_creator

__all__ = ["SingleBandMultiModel"]


class SingleBandMultiModel(ImageLinearFit):
    """Class to simulate/reconstruct images in multi-band option. This class calls
    functions of image_model.py with different bands with decoupled linear parameters
    and the option to pass/select different light models for the different bands.

    the class supports keyword arguments 'index_lens_model_list', 'index_source_light_model_list',
    'index_lens_light_model_list', 'index_point_source_model_list', 'index_optical_depth_model_list' in kwargs_model
    These arguments should be lists of length the number of imaging bands available and each entry in the list
    is a list of integers specifying the model components being evaluated for the specific band.

    E.g. there are two bands, and you want to different light profiles being modeled.
    - you define two different light profiles lens_light_model_list = ['SERSIC', 'SERSIC']
    - set index_lens_light_model_list = [[0], [1]]
    - (optional) for now all the parameters between the two light profiles are independent in the model. You have
    the possibility to join a subset of model parameters (e.g. joint centroid). See the Param() class for documentation.
    """

    def __init__(
        self,
        multi_band_list,
        kwargs_model,
        likelihood_mask_list=None,
        band_index=0,
        kwargs_pixelbased=None,
        linear_solver=False,
    ):
        """

        :param multi_band_list: list of imaging band configurations [[kwargs_data, kwargs_psf, kwargs_numerics],[...], ...]
        :param kwargs_model: model option keyword arguments
        :param likelihood_mask_list: list of likelihood masks (booleans with size of the individual images
        :param band_index: integer, index of the imaging band to model
        :param kwargs_pixelbased: keyword arguments with various settings related to the pixel-based solver
         (see SLITronomy documentation)
        :param linear_solver: bool, if True (default) fixes the linear amplitude parameters 'amp' (avoid sampling) such
         that they get overwritten by the linear solver solution.
        """
        self.type = "single-band-multi-model"
        if likelihood_mask_list is None:
            likelihood_mask_list = [None for _ in range(len(multi_band_list))]
        (
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
        ) = class_creator.create_class_instances(band_index=band_index, **kwargs_model)
        kwargs_data = multi_band_list[band_index][0]
        kwargs_psf = multi_band_list[band_index][1]
        kwargs_numerics = multi_band_list[band_index][2]
        data_i = ImageData(**kwargs_data)
        psf_i = PSF(**kwargs_psf)

        index_lens_model_list = kwargs_model.get(
            "index_lens_model_list", [None for _ in range(len(multi_band_list))]
        )
        self._index_lens_model = index_lens_model_list[band_index]

        index_source_list = kwargs_model.get(
            "index_source_light_model_list", [None for _ in range(len(multi_band_list))]
        )
        self._index_source = index_source_list[band_index]
        index_lens_light_list = kwargs_model.get(
            "index_lens_light_model_list", [None for _ in range(len(multi_band_list))]
        )
        self._index_lens_light = index_lens_light_list[band_index]
        index_point_source_list = kwargs_model.get(
            "index_point_source_model_list", [None for _ in range(len(multi_band_list))]
        )
        self._index_point_source = index_point_source_list[band_index]
        index_optical_depth = kwargs_model.get(
            "index_optical_depth_model_list",
            [None for _ in range(len(multi_band_list))],
        )
        self._index_optical_depth = index_optical_depth[band_index]

        super(SingleBandMultiModel, self).__init__(
            data_i,
            psf_i,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
            kwargs_numerics=kwargs_numerics,
            likelihood_mask=likelihood_mask_list[band_index],
            kwargs_pixelbased=kwargs_pixelbased,
            linear_solver=linear_solver,
        )

    def image(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
        unconvolved=False,
        source_add=True,
        lens_light_add=True,
        point_source_add=False,
    ):
        """Make an image with a realisation of linear parameter values "param".

        :param kwargs_lens: list of keyword arguments corresponding to the superposition
            of different lens profiles
        :param kwargs_source: list of keyword arguments corresponding to the
            superposition of different source light profiles
        :param kwargs_lens_light: list of keyword arguments corresponding to different
            lens light surface brightness profiles
        :param kwargs_ps: keyword arguments corresponding to "other" parameters, such as
            external shear and point source image positions
        :param unconvolved: if True: returns the unconvolved light distribution (prefect
            seeing)
        :param source_add: if True, compute source, otherwise without
        :param lens_light_add: if True, compute lens light, otherwise without
        :param point_source_add: if True, add point sources, otherwise without
        :return: 2d array of surface brightness pixels of the simulation
        """
        (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
        ) = self.select_kwargs(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_extinction
        )
        return self._image(
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
            kwargs_special=kwargs_special,
            unconvolved=unconvolved,
            source_add=source_add,
            lens_light_add=lens_light_add,
            point_source_add=point_source_add,
        )

    def source_surface_brightness(
        self,
        kwargs_source,
        kwargs_lens=None,
        kwargs_extinction=None,
        kwargs_special=None,
        unconvolved=False,
        de_lensed=False,
        k=None,
        update_pixelbased_mapping=False,
    ):
        """Computes the source surface brightness distribution.

        :param kwargs_source: list of keyword arguments corresponding to the
            superposition of different source light profiles
        :param kwargs_lens: list of keyword arguments corresponding to the superposition
            of different lens profiles
        :param kwargs_extinction: list of keyword arguments of extinction model
        :param unconvolved: if True: returns the unconvolved light distribution (prefect
            seeing)
        :param de_lensed: if True: returns the un-lensed source surface brightness
            profile, otherwise the lensed.
        :param k: integer, if set, will only return the model of the specific index
        :return: 2d array of surface brightness pixels
        """
        kwargs_lens_i, kwargs_source_i, _, _, kwargs_extinction_i = self.select_kwargs(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light=None,
            kwargs_ps=None,
            kwargs_extinction=kwargs_extinction,
        )
        return self._source_surface_brightness(
            kwargs_source_i,
            kwargs_lens_i,
            kwargs_extinction=kwargs_extinction_i,
            kwargs_special=kwargs_special,
            unconvolved=unconvolved,
            de_lensed=de_lensed,
            k=k,
            update_pixelbased_mapping=update_pixelbased_mapping,
        )

    def lens_surface_brightness(self, kwargs_lens_light, unconvolved=False, k=None):
        """Computes the lens surface brightness distribution.

        :param kwargs_lens_light: list of keyword arguments corresponding to different
            lens light surface brightness profiles
        :param unconvolved: if True, returns unconvolved surface brightness (perfect
            seeing), otherwise convolved with PSF kernel
        :return: 2d array of surface brightness pixels
        """
        _, _, kwargs_lens_light_i, kwargs_ps_i, _ = self.select_kwargs(
            kwargs_lens=None,
            kwargs_source=None,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_ps=None,
            kwargs_extinction=None,
        )
        return self._lens_surface_brightness(
            kwargs_lens_light_i, unconvolved=unconvolved, k=k
        )

    # def point_source(
    #    self,
    #    kwargs_ps,
    #    kwargs_lens=None,
    #    kwargs_special=None,
    #    unconvolved=False,
    #    k=None,
    # ):
    #    """Computes the point source positions and paints PSF convolutions on them.

    #    :param kwargs_ps:
    #    :param kwargs_lens:
    #    :param kwargs_special:
    #    :param unconvolved:
    #    :param k:
    #    :return:
    #    """
    #    kwargs_lens_i, _, _, kwargs_ps_i, _ = self.select_kwargs(
    #        kwargs_lens=kwargs_lens,
    #        kwargs_source=None,
    #        kwargs_lens_light=None,
    #        kwargs_ps=kwargs_ps,
    #        kwargs_extinction=None,
    #    )
    #    return self._point_source(
    #        kwargs_ps=kwargs_ps_i,
    #        kwargs_lens=kwargs_lens_i,
    #        kwargs_special=kwargs_special,
    #        unconvolved=unconvolved,
    #        k=k,
    #    )

    def likelihood_data_given_model(
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
        """Computes the likelihood of the data given a model. This is specified with the
        non-linear parameters and a linear inversion and prior marginalisation.

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :param check_positive_flux: bool, if True, checks whether the linear inversion
            resulted in non-negative flux components and applies a punishment in the
            likelihood if so.
        :return: log likelihood (natural logarithm) (sum of the log likelihoods of the
            individual images)
        """
        # generate image
        (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
        ) = self.select_kwargs(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_extinction
        )
        logL, param = self._likelihood_data_given_model(
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
            kwargs_special,
            source_marg=source_marg,
            linear_prior=linear_prior,
            check_positive_flux=check_positive_flux,
            linear_solver=self._linear_solver,
        )
        return logL, param

    def error_response(self, kwargs_lens, kwargs_ps, kwargs_special):
        """Returns the 1d array of the error estimate corresponding to the data
        response.

        :return: 1d numpy array of response, 2d array of additional errors (e.g. point
            source uncertainties)
        """
        (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
        ) = self.select_kwargs(
            kwargs_lens,
            kwargs_source=None,
            kwargs_lens_light=None,
            kwargs_ps=kwargs_ps,
            kwargs_extinction=None,
        )
        return self._error_response(
            kwargs_lens_i, kwargs_ps_i, kwargs_special=kwargs_special
        )

        # def extinction_map(self, kwargs_extinction=None, kwargs_special=None):
        """Differential extinction per pixel.

        :param kwargs_extinction: list of keyword arguments corresponding to the optical
            depth models tau, such that extinction is exp(-tau)
        :param kwargs_special: keyword arguments, additional parameter to the extinction
        :return: 2d array of size of the image
        """
        _, _, _, _, kwargs_extinction_i = self.select_kwargs(
            kwargs_extinction=kwargs_extinction
        )
        return self._extinction_map(kwargs_extinction_i, kwargs_special)

    def select_kwargs(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_ps=None,
        kwargs_extinction=None,
        kwargs_special=None,
    ):
        """Select subset of kwargs lists referenced to this imaging band.

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """
        if self._index_lens_model is None or kwargs_lens is None:
            kwargs_lens_i = kwargs_lens
        else:
            kwargs_lens_i = [kwargs_lens[k] for k in self._index_lens_model]
        if self._index_source is None or kwargs_source is None:
            kwargs_source_i = kwargs_source
        else:
            kwargs_source_i = [kwargs_source[k] for k in self._index_source]
        if self._index_lens_light is None or kwargs_lens_light is None:
            kwargs_lens_light_i = kwargs_lens_light
        else:
            kwargs_lens_light_i = [kwargs_lens_light[k] for k in self._index_lens_light]
        if self._index_point_source is None or kwargs_ps is None:
            kwargs_ps_i = kwargs_ps
        else:
            kwargs_ps_i = [kwargs_ps[k] for k in self._index_point_source]
        if self._index_optical_depth is None or kwargs_extinction is None:
            kwargs_extinction_i = kwargs_extinction
        else:
            kwargs_extinction_i = [
                kwargs_extinction[k] for k in self._index_optical_depth
            ]
        return (
            kwargs_lens_i,
            kwargs_source_i,
            kwargs_lens_light_i,
            kwargs_ps_i,
            kwargs_extinction_i,
        )
