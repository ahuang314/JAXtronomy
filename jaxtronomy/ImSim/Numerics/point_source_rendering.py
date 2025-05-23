from jaxtronomy.Util import image_util, kernel_util, util

from functools import partial
from jax import jit, numpy as jnp

__all__ = ["PointSourceRendering"]


class PointSourceRendering(object):
    """Numerics to compute the point source response on an image."""

    def __init__(self, pixel_grid, supersampling_factor, psf):
        """

        :param pixel_grid: PixelGrid() instance
        :param supersampling_factor: int, factor of supersampling of point source
         if None, then uses the supersampling factor of the original PSF
        :param psf: PSF() instance
        """
        self._pixel_grid = pixel_grid
        self._nx, self._ny = self._pixel_grid.num_pixel_axes
        if supersampling_factor is None:
            supersampling_factor = psf.point_source_supersampling_factor
        self._supersampling_factor = supersampling_factor
        self._psf = psf

        # PSF updates are not allowed in jaxtronomy so this can be put in the init
        self._kernel_supersampled = self._psf.kernel_point_source_supersampled(
            self._supersampling_factor, updata_cache=False
        )

    @partial(jit, static_argnums=0)
    def point_source_rendering(self, ra_pos, dec_pos, amp):
        """

        :param ra_pos: list of RA positions of point source(s)
        :param dec_pos: list of DEC positions of point source(s)
        :param amp: list of amplitudes of point source(s)
        :return: 2d numpy array of size of the image with the point source(s) rendered
        """
        subgrid = self._supersampling_factor
        x_pos, y_pos = util.map_coord2pix(
            ra_pos,
            dec_pos,
            self._pixel_grid._x_at_radec_0,
            self._pixel_grid._y_at_radec_0,
            self._pixel_grid._Ma2pix,
        )
        # translate coordinates to higher resolution grid
        x_pos_subgrid = x_pos * subgrid + (subgrid - 1) / 2.0
        y_pos_subgrid = y_pos * subgrid + (subgrid - 1) / 2.0
        kernel_point_source_subgrid = self._kernel_supersampled
        # initialize grid with higher resolution
        subgrid2d = jnp.zeros((self._nx * subgrid, self._ny * subgrid))
        # add_layer2image
        if len(x_pos) > len(amp):
            raise ValueError(
                "there are %s images appearing but only %s amplitudes provided!"
                % (len(x_pos), len(amp))
            )
        for i in range(len(x_pos)):
            subgrid2d = image_util.add_layer2image(
                subgrid2d,
                x_pos_subgrid[i],
                y_pos_subgrid[i],
                amp[i] * kernel_point_source_subgrid,
            )
        # re-size grid to data resolution
        grid2d = image_util.re_size(subgrid2d, factor=subgrid)
        return grid2d * subgrid**2

    @partial(jit, static_argnums=(0, 5))
    def psf_variance_map(self, ra_pos, dec_pos, amp, data, fix_psf_variance_map=False):
        """Variance of PSF error.

        :param ra_pos: image positions of point sources
        :param dec_pos: image positions of point sources
        :param amp: amplitude of modeled point sources
        :param data: 2d numpy array of the data
        :param fix_psf_variance_map: bool, if True, estimates the error based on the
            input (modeled) amplitude, else uses the data to do so.
        :return: 2d array of size of the image with error terms (sigma**2) expected from
            inaccuracies in the PSF modeling
        """
        x_pos, y_pos = util.map_coord2pix(
            ra_pos,
            dec_pos,
            self._pixel_grid._x_at_radec_0,
            self._pixel_grid._y_at_radec_0,
            self._pixel_grid._Ma2pix,
        )
        psf_kernel = self._psf.kernel_point_source
        psf_variance_map = self._psf.psf_variance_map
        variance_map = jnp.zeros_like(data)
        for i in range(len(x_pos)):
            if fix_psf_variance_map is True:
                amp_estimated = amp
            else:
                amp_estimated = kernel_util.estimate_amp(
                    data, x_pos[i], y_pos[i], psf_kernel
                )
            variance_map = image_util.add_layer2image(
                variance_map, x_pos[i], y_pos[i], psf_variance_map * amp_estimated**2
            )
        return variance_map
