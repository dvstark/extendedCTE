# Various programs used in findsources_segmentation.ipynb
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
 
from photutils.utils import calc_total_error, CutoutImage
from photutils.segmentation import detect_sources, deblend_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.background import Background2D
from photutils.aperture import EllipticalAperture

from astropy.convolution import convolve, RickerWavelet2DKernel
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, bootstrap
from astropy.wcs import WCS
from astropy.table import Table, hstack
from astropy.nddata import Cutout2D
from astropy.table import vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

from drizzlepac import pixtopix

from pathlib import Path

import glob
from astropy.table import Table
import numpy as np
from astropy.io import fits

from scipy.stats import spearmanr, theilslopes  


plt.ion()

def calc_background_map(image, background_box_size=512, plot=False,
                        segment_map=None, fig_title=''):
    ''' Program to estimate the background in an image.
    
    The code operates in three steps:
    (1) Find the background using sigma clipping with sources present
    (2) Run source identification
    (3) Rerun background calculation but with sources masked
    
    Alternatively a source mask can be supplied and the first two steps are 
    skipped
    '''
    
    # define a coverage mask for the image
    coverage_mask = ~np.isfinite(image)
    
    if segment_map is not None:
        
        print('Segment map provided, skipping source detection step for \
              background estimation')
        
    else:
        
        # first calc of the background
        print('Estimating background with sources unmasked')
        bkg_init = Background2D(image.astype(float), background_box_size,
                        filter_size=(3, 3), coverage_mask=~np.isfinite(image))
        
        # subtract this background
        image_bkgsub = image - bkg_init.background
        
        # convolve prior to source detection
        print('Detecting and masking sources')
        kernel = make_2dgaussian_kernel(4, size=7)  
        image_bkgsub_convolved = convolve(image_bkgsub, kernel, mask=coverage_mask)

        threshold = 1.5
        segment_map = detect_sources(image_bkgsub_convolved, threshold*bkg_init.background_rms,
                                    npixels=25, mask=coverage_mask)
    
    
    # define mask to block sources
    source_mask = segment_map.data > 0
        
    # recalculate the background, but mask the sources now
    print('Calculating background with sources masked')
    bkg = Background2D(image.astype(float), background_box_size,
                       filter_size=(3, 3), mask=source_mask, 
                       coverage_mask=coverage_mask)
    
    if plot:
        fig, [ax1, ax2] = plt.subplots(1,2, figsize=(15,7))
    
        __, median, std = sigma_clipped_stats(image)
        ax1.imshow(image, origin='lower', aspect='auto', vmin=median - std, vmax=median+5*std, cmap='viridis')
        ax1.set_title(fig_title)
        
        pl=ax2.imshow(bkg.background, origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(pl)
        ax2.set_title('background')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
    return bkg
    

def measure_source_properties(image, header, background_box_size=512,
                              kernel_fwhm=4, kernel_size=7, threshold=2,
                              npixels=25, deblend=True, plot=True, 
                              deblend_nlevels=32, deblend_contrast=0.001, 
                              input_segment_map=None, input_detection_cat=None,
                              localbkg_width=32):
    
    '''wrapper to find sources and measure their properties from star to finish.
    An input catalog and segmentation map can be provided to do matched
    photometry with another image
    
    Steps are:
    1) estimate and subtract background
    2) Convolve data with Gaussian filter
    3) Run source detection (if no input catalog given)
    4) Create source catalog

    '''
    
    # check deblend parameter
    if deblend & (input_segment_map is not None):
        deblend=False
        print('Deblend set to True but input segment map provided. Turning off\
            deblending. Run deblending earlier if it is desired.')
    
    # define wcs of input image
    wcs = WCS(header)
    
    filename = header['FILENAME']
    
    # subtract the background
    bkg = calc_background_map(image, plot=plot, segment_map=input_segment_map,
                              background_box_size=background_box_size,
                              fig_title=filename)
    image_bkgsub = image - bkg.background
    
    # get the total error image
    image_err = calc_total_error(image_bkgsub, bkg.background_rms, 
                                  header['EXPTIME'])
    
    # define smoothing kernel and convolve the data
    kernel = make_2dgaussian_kernel(kernel_fwhm, size=kernel_size)
    image_conv = convolve(image_bkgsub, kernel, mask=~np.isfinite(image_bkgsub))

    # identify sources
    if input_segment_map is None:
        print('Detecting Sources')
        segment_map = detect_sources(image_conv, threshold*bkg.background_rms,
                                    npixels=npixels,
                                    mask=~np.isfinite(image_conv))
    else:
        print('Sources already provided. Skipping source detection.')
        segment_map = input_segment_map
    
    # deblend if desired
    if deblend:
        segment_map = deblend_sources(image_conv, segment_map, npixels=npixels,
                                      nlevels=deblend_nlevels, 
                                      contrast=deblend_contrast)
    
    
    # get catalog from detection image
    if input_detection_cat is not None:
        print('Using input detection catalog for property measurement')
    cat = SourceCatalog(image_bkgsub, segment_map, convolved_data=image_conv,
                        error=image_err, mask=~np.isfinite(image_bkgsub), 
                        background=bkg.background, wcs=wcs, localbkg_width=localbkg_width,
                        apermask_method='mask', kron_params=(2.,2.,4.0),
                        detection_cat=input_detection_cat)
                        # assume default Kron factor is 2.5, like SExtractor
    
    
    # get the masked and unmasked area inside each aperture
    kron_aperture_area = []
    kron_aperture_area_unmasked = []
    for aperture in cat.kron_aperture:
        if aperture is not None:
            kron_aperture_area_unmasked.append(aperture.area_overlap(image_bkgsub,
                                               mask=np.isnan(image_bkgsub)))
            kron_aperture_area.append(aperture.area)
        else:
            kron_aperture_area.append(np.nan)
            kron_aperture_area_unmasked.append(np.nan)
            
    
    cat.add_extra_property('kron_area', np.array(kron_aperture_area))
    cat.add_extra_property('kron_area_unmasked', np.array(kron_aperture_area_unmasked))

    # plot the results of the detection if set to True
    if plot:
        
        fig, [ax1, ax2] = plt.subplots(1,2,figsize=(15, 7))
        __, __, std = sigma_clipped_stats(image_bkgsub)
        ax1.imshow(image_bkgsub, origin='lower', aspect='auto', vmin=-std, vmax=3*std, cmap='Greys')
        cat.plot_kron_apertures(ax=ax1, color='red', lw=0.75)
        ax1.set_title('Image w/ Kron Apertures')
        
        ax2.imshow(segment_map, origin='lower', aspect='auto', interpolation='nearest', cmap=segment_map.cmap)
        #cat.plot_kron_apertures(ax=ax2, color='red', lw=0.75)
        ax2.set_title('Segment Map')
    
    return cat, segment_map, bkg

def get_star_probs(image_file, photutils_catalog, 
               sex_file='/Users/dstark/acs_work/cte/extended/config.sex',
               param_file='/Users/dstark/acs_work/cte/extended/default.param',
               pxscl=0.03333):   
    
    # run SExtractor first
    command = 'sex {} -c {} -parameters_name {} -weight_type NONE'.format(image_file, sex_file, param_file)
    subprocess.call(command.split())    
    
    # load the SExtractor table
    se_cat = Table.read('sources.cat', format='ascii.sextractor')
    se_coords = SkyCoord(ra=se_cat['ALPHA_J2000'], dec=se_cat['DELTA_J2000'])

    # set up coordinate objects from main catalog
    det_coords = photutils_catalog.sky_centroid
    # some NaNs need to be addressed...
    sel = ~np.isfinite(det_coords.ra) | ~np.isfinite(det_coords.dec)
    det_coords[sel] = SkyCoord(ra=0 * u.deg, dec = -90*u.deg, frame='fk5')
    
    # run crossmatch
    idx, d2d, d3d = det_coords.match_to_catalog_sky(se_coords)
    sel=d2d.arcsec/pxscl <= 1 # matches wtihin 1 pixel
    photutils_catalog.add_extra_property('star_prob', np.zeros(len(photutils_catalog)).astype(float))
    photutils_catalog.star_prob[sel] = se_cat['CLASS_STAR'][idx[sel]]
    
    return photutils_catalog
    
def create_exposure_dictionary(dir, quiet=True):

    '''creates a dictionary of information about each exposure. This is highly 
    customized and not for general use'''

    # create exposure dictionary
    input_images = np.array(glob.glob('{}/**/*_fl[c].fits'.format(dir), recursive=True))
    image_types = np.array(['flc']*len(input_images))
    image_types = np.concatenate([image_types, np.array(['flt']*len(input_images))])
    input_images = np.concatenate([input_images, np.array(glob.glob('{}/**/*_flt.fits'.format(dir), recursive=True))])

    if not quiet:
        print(len(input_images), 'images found')
        print(len(image_types), 'image types found')
        print(input_images)

    asn_ids = []
    prop_ids = []
    postarg1 = []
    postarg2 = []
    pa_v3 = []
    
    for file in input_images:
        with fits.open(file) as hdu:
            hdr0 = hdu[0].header
            asn_ids.append(hdr0['ASN_ID'])
            prop_ids.append(hdr0['PROPOSID'])
            postarg1.append(hdr0['POSTARG1'])
            postarg2.append(hdr0['POSTARG2'])
            pa_v3.append(hdr0['PA_V3'])

    asn_ids = np.array(asn_ids).astype(str)
    prop_ids = np.array(prop_ids).astype(str)
    input_images = np.array(input_images)

    exps = Table()
    exps['prop_id'] = prop_ids
    exps['asn_id'] = asn_ids
    exps['image_path'] = input_images
    exps['image_type'] = image_types
    exps['postarg1'] = np.array(postarg1)
    exps['postarg2'] = np.array(postarg2)
    exps['pa_v3'] = np.array(pa_v3)

    # get the roots out of the input_images arrays for convenience
    roots = []
    for exp in exps:
        root = exp['image_path'].split('/')[-1].split('_')[0]
        roots.append(root)

    exps['root'] = roots
    
    return exps
    
def add_detector_pos(sources, measurement_image, flc_images, plot=False):

    ''' Adds the detector position (in flc/flt) to photutils catalog of
    sources. Since there can be more than one flc/flt exposure for a given 
    image, it looks at all x/y detector positions and then averages them.'''

    counter = 0
    for f in flc_images:
        x_det, y_det, chip = get_wfc_coords(measurement_image, f, sources.xcentroid, sources.ycentroid, plot=plot)

        if counter == 0:
            x_det_all = x_det
            y_det_all = y_det
            chip_all = chip
        else:
            x_det_all = np.vstack([x_det_all, x_det])
            y_det_all = np.vstack([y_det_all, y_det])
            chip_all = np.vstack([chip_all, chip])

        counter += 1

    # now take mean of all non-nan entries in x_det_all
    x_det_all[x_det_all == -9999] = np.nan
    y_det_all[y_det_all == -9999] = np.nan

    x_det_final = np.nanmean(x_det_all,axis=0)
    y_det_final = np.nanmean(y_det_all, axis=0)

    # remove cases where the source does not fall on all chips
    n_image = np.sum(np.isfinite(x_det_all), axis=0) + np.sum(np.isfinite(y_det_all), axis=0)
    sel = n_image < (2*len(flc_images))
    x_det_final[sel] = np.nan
    y_det_final[sel] = np.nan

    # define y_shifts based on the Y detector coord and chip
    y_shifts = np.zeros_like(x_det_final)
    sel = chip == 1
    y_shifts[sel] = 2047 - y_det_final[sel]
    sel = chip == 2
    y_shifts[sel] = y_det_final[sel]

    sources.x_det = x_det_final
    sources.y_det = y_det_final 
    sources.y_shifts = y_shifts
    sources.chip = chip

    return sources

def get_wfc_coords(drz_image, wfc_image, x, y, return_each_chip=False, plot=False):
    ''' Gets the coordinates in flc/flt corresponding to input position in
    drc/drz. Tries to automatically figure out which chip the source was on. 
    Basically a wrapper for drizzlepac.pixtopix.tran'''

    if plot:
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10,5))

    # try first extension (WFC2)
    x_det1, y_det1 = pixtopix.tran(drz_image, wfc_image + "[SCI, 1]", x=x, y=y, verbose=False, direction='forward')
    print(np.nanmin(y_det1), np.nanmax(y_det1))

    # convert to arrays
    x_det1 = np.array(x_det1)
    y_det1 = np.array(y_det1)

    # indicate which fall off the chip
    sel = (y_det1 < 0) | (y_det1 > 2047) | (x_det1 < 0) | (x_det1 > 4095)
    x_det1[sel] = -9999
    y_det1[sel] = -9999

    if plot:
        sel = y_det1 != -9999
        pl=ax1.scatter(x[sel], y[sel], c=y_det1[sel], cmap='Blues')
        ax1.set_xlabel('x_image')
        ax1.set_ylabel('y_image')
        plt.colorbar(pl, label='y_det')

    # try fourth extension (WFC1)
    x_det4, y_det4 = pixtopix.tran(drz_image, wfc_image + "[SCI, 2]", x=x, y=y, verbose=False, direction='forward')
    print(np.nanmin(y_det4), np.nanmax(y_det4))

    # convert to arrays
    x_det4 = np.array(x_det4)
    y_det4 = np.array(y_det4)

    # indicate which fall off the chip
    sel = (y_det4 < 0) | (y_det4 > 2047) | (x_det4 < 0) | (x_det4 > 4095)
    x_det4[sel] = -9999
    y_det4[sel] = -9999

    if plot:
        sel = y_det4 != -9999
        pl=ax2.scatter(x[sel], y[sel], c=y_det4[sel], cmap='Reds')
        ax2.set_xlabel('x_image')
        ax2.set_ylabel('y_image')
        plt.colorbar(pl, label='y_det')

    # now, figure out which chip. Should be -9999 in both x and  y
    chip = np.zeros(len(x))
    chip[(x_det1 > -9999) & (y_det1 > -9999)] = 2
    chip[(x_det4 > -9999) & (y_det4 > -9999)] = 1

    x_det = np.zeros_like(x_det1)-9999.
    y_det = np.zeros_like(x_det1)-9999.
    
    sel = (chip == 2)
    x_det[sel] = x_det1[sel]
    y_det[sel] = y_det1[sel]

    sel = (chip == 1)
    x_det[sel] = x_det4[sel]
    y_det[sel] = y_det4[sel]

    if not return_each_chip:
        return x_det, y_det, chip
    else:
        return x_det, y_det, chip, x_det4, y_det4, x_det1, y_det1
    
    


def add_background_levels(sources, ref_images):
    
    '''Adds the individual exposure backgrounds due to sky, dark, and flash to 
    a photutils catalog'''
    
    n = len(sources)
    sources.background_image = np.array([0.] * n)
    sources.background_flash = np.array([0.] * n)
    sources.background_dark = np.array([0.] * n)

    # Do WFC1 first, extension 4
    sel = (sources.chip == 1) & (sources.x_det >= 0) & (sources.x_det < 4096) & (sources.y_det >= 0) & (sources.y_det < 2048) 
    bgfiles = [image.replace('.fits','_bkg_ext4.fits') for image in ref_images]
    flashval, darkval, imval = get_background_levels(sources.x_det[sel], sources.y_det[sel], 'WFC1', ref_images[0], bgfiles=bgfiles)
    sources.background_image[sel] = imval
    sources.background_flash[sel] = flashval
    sources.background_dark[sel] = darkval

    # next is WFC2, extension 1
    sel = (sources.chip == 2) & (sources.x_det >= 0) & (sources.x_det < 4096) & (sources.y_det >= 0) & (sources.y_det < 2048) 
    bgfiles = [image.replace('.fits','_bkg_ext1.fits') for image in ref_images]
    flashval, darkval, imval = get_background_levels(sources.x_det[sel], sources.y_det[sel], 'WFC2', ref_images[0], bgfiles=bgfiles)
    sources.background_image[sel] = imval
    sources.background_flash[sel] = flashval
    sources.background_dark[sel] = darkval 

    sources.background_total = sources.background_image + sources.background_flash + sources.background_dark
    
    return sources   

def get_background_levels(xdet, ydet, chip, fl_file, bgfiles=None):

    '''Retrieves the individual expoure backgrounds from sky, dark, and flash
    for a given position'''

    os.environ['CRDS_PATH'] = os.environ['HOME'] + '/crds_cache'
    os.environ['jref'] = os.environ['CRDS_PATH'] + "/references/hst/jref/"
    jref = os.environ['jref']

    # load the image
    hdu = fits.open(fl_file)

    # define the extension from the chip number
    if chip == 'WFC1':
        ext = 4
    else:
        ext = 1

    hdr = hdu[0].header
    image = hdu[ext].data
    hdu.close()

    

    # flash info
    flashfile = hdr['FLSHFILE'].split('$')[-1]
    flashdur = hdr['FLASHDUR']
    print('flash file = ' + flashfile)
    
    # dark info
    DRKCFILE = hdr['DRKCFILE'].split('$')[-1]
    DARKFILE = hdr['DARKFILE'].split('$')[-1]
    if 'flc' in fl_file:
        final_dkfile = DRKCFILE
    else:
        final_dkfile = DARKFILE

    print('Dark file: ' + final_dkfile)
    EXPTIME = hdr['EXPTIME']  

    # get if it is already cahced
    if (not Path(jref + flashfile).exists()) & (flashfile != 'N/A'):
        get_flash = os.system('crds sync --hst --files ' + flashfile + ' --output-dir '+os.environ['jref'])

    if not Path(jref + final_dkfile).exists():
        get_dark = os.system('crds sync --hst --files ' + final_dkfile + ' --output-dir '+os.environ['jref'])


    # open the flash reference file
    if flashfile != 'N/A':
        fhdu = fits.open(jref + flashfile)
        flash_image = fhdu[ext].data
        fhdu.close()

        # round xdet and ydet and get corresponding values
        flashval_ref = np.array([flash_image[y.astype(int), x.astype(int)] for x, y in zip(xdet, ydet)])

        # scale the flashval by the flashduration
        flashval = flashval_ref * flashdur
    else:
        flashval = np.zeros_like(xdet)

    print('range of flash levels: ', np.min(flashval), np.max(flashval))

    # open the dark reference image
    dhdu = fits.open(jref + final_dkfile)
    dark_image = dhdu[ext].data
    dhdu.close()
    
    # do the same for the dark value
    #darkval_ref = np.array([dark_image[y.astype(int), x.astype(int)] for x, y in zip(xdet, ydet)])
    # simplifying to a single value due to strong variations in dark value...need to consider how to handle those
    darkval_ref = np.array([np.median(dark_image)]*len(xdet))

    # scale by exposure time + flashdur (there's some extra overhead that needs to be added here!)
    darkval = darkval_ref * (EXPTIME + flashdur)

    # do the image itself now, unless the bgfiles are provided, in which case, use those
    if bgfiles is None:
        __, median, __ = sigma_clipped_stats(image)
        imval = np.array([median]*len(xdet))
    else:
        # load the bg files
        bg_images = []
        for bgfile in bgfiles:
            bg_hdu = fits.open(bgfile)
            bg_images.append(bg_hdu[0].data)
            bg_hdu.close()
        # get the median of the background images
        bg_images = np.array(bg_images)
        med_bg = np.nanmedian(bg_images, axis=0)

        # get value at each position
        imval = np.array([med_bg[y.astype(int), x.astype(int)] for x, y in zip(xdet, ydet)])


    return flashval, darkval, imval
    
    
def add_additional_properties(cat, radii=None):
    '''Code to add custom properties to a photutils source catalog'''


    # segment surface brightness
    segment_surface_brightness = cat.segment_flux / cat.area
    cat.add_extra_property('segment_surface_brightness', segment_surface_brightness)
    
    # kron surface brightness
    b_a = 1 / cat.elongation
    kron_area = np.pi * cat.kron_radius**2 * b_a
    kron_surface_brightness = cat.kron_flux / kron_area
    cat.add_extra_property('kron_surface_brightness', kron_surface_brightness)
    
    # scaled kron radius in pixels
    kron_scaled = cat.kron_params[0] * cat.kron_radius
    cat.add_extra_property('kron_radius_scaled', kron_scaled)
    
    # get kron flux at 1x the scale length (so this would be kron_radius * semimaj * 1
    cat.kron_photometry([1,2], name='half_kron', overwrite=False)
    
    cat.kron_photometry([0.5,2], name='quarter_kron', overwrite=False)
    
    # add in circular and elliptical aperture photometry at specified radii, if any
    if radii is not None:
        for rad in radii:
            print('running circular aperture photometry w/ radius = {} pix'.format(rad))
            cat.circular_photometry(rad, name='circ_flux_{}pix'.format(rad))
            
        

    
    return cat



def dmag_background(flux_var, xvar_name, image_type, min_yshifts=0, max_yshifts=9999,
                    nbins=4, binmin=None, binmax=None, ylim=[-0.5,0.5], xlim=None,
                    logxvar = False, use_det_xvar=False, max_kron_rad=9999, 
                    min_mag = -9999, max_mag = 26, cvar_name = 'background_total',
                    cvar_binmin=15, cvar_binmax=40, cvar_nbins=8, fit_line=False,
                    return_outliers=True, savefig=None, catalog_suffix=None,
                    title_suffix=None, bin_shift_mag=0.05, min_det_snr=10,
                    min_meas_snr = 5, deblend=False, nolocalbkg=False, altbg=False,
                    cvar_cmap='plasma', axes=None):
    
    '''Plots delta_mag vs another property, broken up by background level'''
    if altbg:
        alt_bg_str = '_bkg256'
    else:
        alt_bg_str = ''
    if deblend:
        deblend_str = '_deblend'
    else:
        deblend_str = ''
        
    if nolocalbkg:
        bkg_str='_nolocalbkg'
    else:
        bkg_str=''
    
    # get the exposure dictionary
    exps = create_exposure_dictionary()
    
    prop_ids = ['13603', '16870']  # later epoch data

    # define the flux error variable name
    flux_err_var = flux_var.replace('flux', 'fluxerr')


    # create a template to hold the statistical info
    first_time = True
    row = Table()

    # set up the plot grid, if needed
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize = (10, 4), sharey=True)

    ### read the detection image catalog ###
    source_table_det = f'/Users/dstark/acs_work/cte/extended/data/processed/10325/10325_J91JC4010_drc_sci{deblend_str}{bkg_str}{alt_bg_str}_photutils_cat.ecsv'
    det_cat = Table.read(source_table_det)
    det_image = source_table_det.split('sci')[0] + 'sci.fits'
    det_photflam = fits.getval(det_image,'PHOTFLAM')
    print(f'det_photflam = {det_photflam}')
    det_cat['MAG'] = -2.5*np.log10(det_cat[flux_var]*det_photflam) - 21.10
    
    # keep track of the length of the detection catalog
    det_cat_n = len(det_cat)

    # define some quality selections
    det_snr = det_cat[flux_var] / det_cat[flux_err_var]
    good_det = (det_snr > min_det_snr) & (det_cat['star_prob'] < 0.3) & \
        ((det_cat['kron_radius']*det_cat['semimajor_sigma']/4) > 2) & \
        (det_cat['xcentroid'] < (det_cat['ycentroid']/27.51+6417)) & \
        (det_cat['ycentroid'] > (-0.112*det_cat['xcentroid']+745.2)) & \
        (det_cat['ycentroid'] < (-0.089*det_cat['xcentroid'] + 6741.1)) & \
        (det_cat['xcentroid'] > (det_cat['ycentroid']/109.15+322.4)) & \
        ((det_cat['ycentroid'] > (43.8*det_cat['xcentroid'] - 141451.4)) | (det_cat['ycentroid'] < (44.5*det_cat['xcentroid'] - 155626.5)))
    det_cat['good'] = good_det
    # the coordinate selections remove junk at the edges where the dither patterns don't overlap well and data quality is poor
    
    for i, prop_id in enumerate(prop_ids):
        print(f'On program {prop_id}')

        # get the unique asns
        sel = exps['prop_id'] == prop_id
        asns = np.unique(exps['asn_id'][sel])
        print('Unique ASNs in this program: ', asns)

        # cycle through the asns
        for j, asn in enumerate(asns):
            print(f'Pulling data from prop id {prop_id}, ASN {asn}')

            source_table_meas = '/Users/dstark/acs_work/cte/extended/data/processed/{}/{}_{}_{}_sci_align{}{}{}_photutils_cat.ecsv'.format(prop_id, prop_id, asn, image_type, deblend_str, bkg_str, alt_bg_str)
            sources_meas = Table.read(source_table_meas)
            
            #make sure this has the same length as the detection image
            sources_meas_n = len(sources_meas)
            if (sources_meas_n != det_cat_n) | (np.sum(sources_meas['label'] != det_cat['label']) > 0):
                print('WARNING: DETECTION CATALOG AND MEASUREMENT CATALOGS DO NOT MATCH!')
                return -1

            # add some additional info
            sources_meas['PROP_ID'] = prop_id
            sources_meas['ASN'] = asn

            meas_image = source_table_meas.split('_align')[0] + '.fits'
            meas_photflam = fits.getval(meas_image,'PHOTFLAM')
            print(f'meas_photflam = {meas_photflam}')

            # get magnitudes of detection/image
            sources_meas['MAG'] = -2.5*np.log10(sources_meas[flux_var]*meas_photflam) - 21.10

            # add all the data to a master table. Copy the detection catalog data for convenience
            if j == 0:
                det_cat_full = det_cat.copy()
                meas_cat = sources_meas.copy()
            else:
                det_cat_full = vstack([det_cat_full, det_cat.copy()])
                meas_cat = vstack([meas_cat, sources_meas.copy()])

        # define x variable
        xlabel = xvar_name
        xvar = meas_cat[xvar_name]
        if use_det_xvar:
            xvar = det_cat_full[xvar_name]
        
        # log the x var if needed and adjust the label
        if logxvar:
            xvar = np.log10(xvar)
            xlabel = 'log ' + xlabel

        # set up the binned value scheme
        pc = np.percentile(xvar[np.isfinite(xvar)], [1,99])
        if binmin is None:
            binmin = pc[0]
        if binmax is None:
            binmax = pc[1]

        bins = np.linspace(binmin,binmax,nbins+1)
        binsmin = bins[:-1]
        binsmax = bins[1:]
        binsize = binsmax[0] - binsmin[0]

        # define magnitude change
        dmag = det_cat_full['MAG'] - meas_cat['MAG']  # this is negative if we lose flux in later epochs

        # indicate which sources to consider
        snr_meas = meas_cat[flux_var] / meas_cat[flux_err_var]

        # fig2, [pax1, pax2] = plt.subplots(1,2, figsize=(12,5))
        # pax1.hist((meas_cat['area']/meas_cat['segment_area']),bins=100)     
        # pax2.hist((meas_cat['kron_area_unmasked'] / meas_cat['kron_area']), bins=100)   

        # mi = fits.getdata(meas_image)
        
        # fig, tax = plt.subplots()
        # tax.imshow(mi, origin='lower', aspect='auto')
        # sel=(meas_cat['kron_area_unmasked'] / meas_cat['kron_area']) < 0.9
        # tax.scatter(meas_cat['xcentroid'][sel], meas_cat['ycentroid'][sel], alpha=0.5, s=5, zorder=2)
        # sel=(meas_cat['kron_area_unmasked'] / meas_cat['kron_area']) > 0.9
        # tax.scatter(meas_cat['xcentroid'][sel], meas_cat['ycentroid'][sel], alpha=0.5, s=5)
        
        good = det_cat_full['good'] & \
            ((meas_cat['kron_radius']*meas_cat['semimajor_sigma']) > 2) & \
            (meas_cat['y_shifts'] > min_yshifts) & \
            (meas_cat['y_shifts'] < max_yshifts) & \
            (np.isfinite(meas_cat['y_shifts'])) & \
            (snr_meas > min_meas_snr) & \
            (np.isfinite(xvar)) & \
            (meas_cat['MAG'] <= max_mag) & \
            (meas_cat['MAG'] >= min_mag) & \
            ((meas_cat['area']/meas_cat['segment_area']) > 0.99) & \
            ((meas_cat['kron_area_unmasked'] / meas_cat['kron_area']) > 0.99) 

        # (det_cat['CLASS_STAR'] < 0.05) & (sources['FLAGS'] == 0)  & (det_cat['FLAGS'] <= 2) & (det_cat['KRON_RADIUS'] > 3.5)  & (det_cat['KRON_RADIUS'] < max_kron_rad)
            
        ax = axes[i]
        sel = good
        print(f'color variable range: {np.min(meas_cat[cvar_name][sel])} - {np.max(meas_cat[cvar_name][sel])}')
        if not means_only:
            pl=ax.scatter(xvar[sel], dmag[sel], s=3, c=meas_cat[cvar_name][sel], cmap=cvar_cmap, alpha=0.5)
            plt.colorbar(pl, label='background [e-/pix]')

        print(binmin,binmax, binsize)
        if xlim is None:
            ax.set_xlim(binmin-binsize/2,binmax+binsize/2)
        else:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.axhline(0,lw=3, ls='--', color='red', alpha=0.5)

        if prop_id == '13603': 
            ax.text(0.01,0.99,'2013',ha='left', va='top', transform=ax.transAxes, fontsize='x-large')
            #ax.set_title('2013')
        else:
            #ax.set_title('2021')
            ax.text(0.01,0.99,'2021',ha='left', va='top', transform=ax.transAxes, fontsize='x-large')

        ax.set_ylabel(r'$\Delta mag$ (truth - measured)')
        ax.set_xlabel(xlabel)

        # set up the secondary bins
        cvar_pc = np.percentile(meas_cat[cvar_name][np.isfinite(meas_cat[cvar_name])],[1,99])
        if cvar_binmin is None:
            cvar_binmin = cvar_pc[0] #np.nanmin(meas_cat[cvar_name])
        if cvar_binmax is None:
            cvar_binmax = cvar_pc[1] #np.nanmax(meas_cat[cvar_name])

        cvar_bins = np.linspace(cvar_binmin,cvar_binmax,cvar_nbins+1)
        cvar_binsmin = cvar_bins[:-1]
        cvar_binsmax = cvar_bins[1:]

        plt.set_cmap('Set2')

        bin_xshift = np.linspace(-1,1,len(cvar_binsmin))*binsize*bin_shift_mag
        color_set = ['#e41a1c','#377eb8', '#4daf4a', 'magenta']
        marker_set = ['o','v','P', 'D']
                
        for cmin, cmax, color, shift, marker in zip(cvar_binsmin, cvar_binsmax, color_set, bin_xshift, marker_set):
            bin_mean = []
            bin_median = []
            bin_mean_err = []
            bin_std = []
            bin_n = []
            for lo, hi in zip(binsmin, binsmax):
                print(lo, hi)
                sel=good & (xvar >= lo) & (xvar < hi) & (meas_cat[cvar_name] >= cmin) & (meas_cat[cvar_name] < cmax) & (np.isfinite(dmag))
                mean, median, std = sigma_clipped_stats(dmag[sel])
                boots = bootstrap(dmag[sel], bootfunc=np.nanmedian, bootnum=1000)
                bin_n.append(np.sum(sel))
                bin_median.append(median)
                bin_mean.append(mean)
                bin_std.append(std) 
                bin_mean_err.append(np.std(boots))
            bin_n = np.array(bin_n)
            bin_mean = np.array(bin_mean)
            bin_std = np.array(bin_std)
            bin_mean_err = np.array(bin_mean_err)
            print('number per bin: ', bin_n)

            ax.errorbar((binsmin + binsmax)/2 + shift, bin_median, bin_mean_err, marker=marker, ls='none', alpha=0.75, color=color, label='{} - {}'.format(cmin, cmax), markersize=9,lw=3)

        ax.legend(title='background (e-/pix)', loc='upper right')
    plot_title = 'Image Type = {}; Flux Type = {}'.format(image_type.upper(), flux_var)
    if title_suffix is not None:
        plot_title = plot_title + ' ;'+title_suffix
    plt.suptitle(plot_title, fontsize='x-large')

    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig, dpi=150)
        
    return axes

def dmag_background2(flux_var, xvar_name, image_type, min_yshifts=0, max_yshifts=9999,
                    nbins=4, binmin=None, binmax=None, ylim=[-0.5,0.5], xlim=None,
                    logxvar = False, use_det_xvar=False, max_kron_rad=9999, 
                    min_mag = 21.5, max_mag = 26, cvar_name = 'background_total',
                    cvar_binmin=15, cvar_binmax=40, cvar_nbins=8, fit_line=False,
                    return_outliers=True, savefig=None, catalog_suffix=None,
                    title_suffix=None, bin_shift_mag=0.1, min_det_snr=10,
                    min_meas_snr = 5, deblend=False, nolocalbkg=False, altbg=False,
                    cvar_cmap='Greys', use_det_cvar=False, verbose=False, nolegend=False,
                    min_meas_background=0, max_meas_background=9999, legend_loc='lower left',
                    scatter_alpha=0.5, cvar_label='Background [e-/pix]', means_only=False,
                    axes=None, mean_alpha=0.75, xlabel=None, ylabel=None,
                    color_set=None, min_xvar=None, max_xvar=None):
    
    print('\nImage Type = {}'.format(image_type))


    '''Plots delta_mag vs another property, broken up by background level'''
    if altbg:
        alt_bg_str = '_bkg256'
    else:
        alt_bg_str = ''
    if deblend:
        deblend_str = '_deblend'
    else:
        deblend_str = ''
        
    if nolocalbkg:
        bkg_str='_nolocalbkg'
    else:
        bkg_str=''
    
    # get the exposure dictionary
    exps = create_exposure_dictionary(quiet=True)
    
    prop_ids = ['13603', '16870']  # later epoch data

    # define the flux error variable name
    flux_err_var = flux_var.replace('flux', 'fluxerr')


    # create a template to hold the statistical info
    first_time = True
    row = Table()

    # set up the plot grid
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize = (10, 4), sharey=True)

    ### read the detection image catalog ###
    source_table_det = f'/Users/dstark/acs_work/cte/extended/data/processed/10325/10325_J91JC4010_drc_sci{deblend_str}{bkg_str}{alt_bg_str}_photutils_cat.ecsv'
    det_cat = Table.read(source_table_det)
    det_image = source_table_det.split('sci')[0] + 'sci.fits'
    det_photflam = fits.getval(det_image,'PHOTFLAM')
    if verbose:
        print(f'det_photflam = {det_photflam}')
    det_cat['MAG'] = -2.5*np.log10(det_cat[flux_var]*det_photflam) - 21.10
    det_cat['MAGERR'] = 2.5/np.log(10)*det_cat[flux_err_var] / det_cat[flux_var]
    det_cat['counts_per_exposure'] = det_cat[flux_var] * 500  # assuming 500 seconds exposure time for the detection image
    det_cat['log_counts_per_exposure'] = np.log10(det_cat['counts_per_exposure'])
 
    det_cat['total_counts_per_exposure'] = det_cat['kron_flux'] * 500  # assuming 500 seconds exposure time for the detection image
    det_cat['log_total_counts_per_exposure'] = np.log10(det_cat['total_counts_per_exposure'])

    det_cat.meta['PHOTFLAM'] = det_photflam

    det_cat['kron_sb'] = np.log10(det_cat['half_kron_flux'] / (np.pi * (det_cat['kron_radius'] * det_cat['semimajor_sigma'])**2/det_cat['elongation']))
    #print('kron_sb_min_max: ', det_cat['kron_sb'].min(), det_cat['kron_sb'].max())
    det_cat['kron_mag'] = -2.5*np.log10(det_cat['kron_flux']*det_photflam) - 21.10

    # keep track of the length of the detection catalog
    det_cat_n = len(det_cat)

    # define some quality selections
    good_det = standard_det_qual(det_cat, min_mag=min_mag, max_mag=max_mag, min_snr = min_det_snr)

    # det_snr = det_cat[flux_var] / det_cat[flux_err_var]
    # good_det = (det_snr > min_det_snr) & (det_cat['star_prob'] < 0.3) & \
    #     ((det_cat['kron_radius']*det_cat['semimajor_sigma']/4) > 2) & \
    #     (det_cat['xcentroid'] < (det_cat['ycentroid']/27.51+6417)) & \
    #     (det_cat['ycentroid'] > (-0.112*det_cat['xcentroid']+745.2)) & \
    #     (det_cat['ycentroid'] < (-0.089*det_cat['xcentroid'] + 6741.1)) & \
    #     (det_cat['xcentroid'] > (det_cat['ycentroid']/109.15+322.4)) & \
    #     ((det_cat['ycentroid'] > (43.8*det_cat['xcentroid'] - 141451.4)) | (det_cat['ycentroid'] < (44.5*det_cat['xcentroid'] - 155626.5)))
    det_cat['good'] = good_det
    # the coordinate selections remove junk at the edges where the dither patterns don't overlap well and data quality is poor
    
    first_time = True

    for i, prop_id in enumerate(prop_ids):
        print(f'\nOn program {prop_id}')

        # get the unique asns
        sel = exps['prop_id'] == prop_id
        asns = np.unique(exps['asn_id'][sel])
        if verbose:
            print('Unique ASNs in this program: ', asns)

        # cycle through the asns
        for j, asn in enumerate(asns):
            if verbose:
                print(f'Pulling data from prop id {prop_id}, ASN {asn}')

            source_table_meas = '/Users/dstark/acs_work/cte/extended/data/processed/{}/{}_{}_{}_sci_align{}{}{}_photutils_cat.ecsv'.format(prop_id, prop_id, asn, image_type, deblend_str, bkg_str, alt_bg_str)
            sources_meas = Table.read(source_table_meas)
            
            #make sure this has the same length as the detection image
            sources_meas_n = len(sources_meas)
            if (sources_meas_n != det_cat_n) | (np.sum(sources_meas['label'] != det_cat['label']) > 0):
                print('WARNING: DETECTION CATALOG AND MEASUREMENT CATALOGS DO NOT MATCH!')
                return -1

            # add some additional info
            sources_meas['PROP_ID'] = prop_id
            sources_meas['ASN'] = asn

            meas_image = source_table_meas.split('_align')[0] + '.fits'
            meas_photflam = fits.getval(meas_image,'PHOTFLAM')
            if verbose:
                print(f'meas_photflam = {meas_photflam}')

            # get magnitudes of detection/image
            sources_meas['MAG'] = -2.5*np.log10(sources_meas[flux_var]*meas_photflam) - 21.10
            sources_meas['MAGERR'] = 2.5/np.log(10)*sources_meas[flux_err_var] / sources_meas[flux_var]

            sources_meas['kron_mag'] = -2.5*np.log10(sources_meas['kron_flux']*meas_photflam) - 21.10

            sources_meas['counts_per_exposure'] = sources_meas[flux_var] * 500  # assuming 500 seconds exposure time for the measurement image
            sources_meas['log_counts_per_exposure'] = np.log10(sources_meas['counts_per_exposure'])
            
            sources_meas['total_kron_counts_per_exposure'] = sources_meas['kron_flux'] * 500  # assuming 500 seconds exposure time for the measurement image
            sources_meas['log_total_kron_counts_per_exposure'] = np.log10(sources_meas['total_kron_counts_per_exposure'])

            sources_meas['mean_counts_per_exposure'] = sources_meas['counts_per_exposure'] / sources_meas['kron_area']  # mean counts per exposure
            
            sources_meas['total_counts_per_exposure'] = sources_meas['counts_per_exposure'] + sources_meas['background_total']*sources_meas['kron_area_unmasked']  # total counts per exposure, including background
            sources_meas['log_total_counts_per_exposure'] = np.log10(sources_meas['total_counts_per_exposure'])

            sources_meas['mean_total_counts_per_exposure'] = sources_meas['mean_counts_per_exposure'] + sources_meas['background_total']  # mean total counts per exposure, including background

            sources_meas['kron_snr'] = sources_meas['kron_flux'] / sources_meas['kron_fluxerr']
            # add all the data to a master table. Copy the detection catalog data for convenience
            if j == 0:
                det_cat_full = det_cat.copy()
                meas_cat = sources_meas.copy()
            else:
                det_cat_full = vstack([det_cat_full, det_cat.copy()])
                meas_cat = vstack([meas_cat, sources_meas.copy()])

        # define x variable
        if xlabel is None:
            xlabel = xvar_name
        if use_det_xvar:
            xvar = det_cat_full[xvar_name]
        else:
            xvar = meas_cat[xvar_name]

        
        # log the x var if needed and adjust the label
        if logxvar:
            xvar = np.log10(xvar)
            xlabel = 'log ' + xlabel

        # set up the binned value scheme
        pc = np.percentile(xvar[np.isfinite(xvar)], [1,99])
        if binmin is None:
            binmin = pc[0]
        if binmax is None:
            binmax = pc[1]

        bins = np.linspace(binmin,binmax,nbins+1)
        print('bins: ', bins)
        binsmin = bins[:-1]
        binsmax = bins[1:]
        binsize = binsmax[0] - binsmin[0]

        # define magnitude change
        #dmag = det_cat_full['MAG'] - meas_cat['MAG']  # this is negative if we lose flux in later epochs
        dmag = meas_cat['MAG'] - det_cat_full['MAG']  # fixed this to use the proper magnitude convention. 

        dmag_err = np.sqrt(det_cat_full['MAGERR']**2 + meas_cat['MAGERR']**2) 

        # indicate which sources to consider
        snr_meas = meas_cat[flux_var] / meas_cat[flux_err_var]

        # fig2, [pax1, pax2] = plt.subplots(1,2, figsize=(12,5))
        # pax1.hist((meas_cat['area']/meas_cat['segment_area']),bins=100)     
        # pax2.hist((meas_cat['kron_area_unmasked'] / meas_cat['kron_area']), bins=100)   

        # mi = fits.getdata(meas_image)
        
        # fig, tax = plt.subplots()
        # tax.imshow(mi, origin='lower', aspect='auto')
        # sel=(meas_cat['kron_area_unmasked'] / meas_cat['kron_area']) < 0.9
        # tax.scatter(meas_cat['xcentroid'][sel], meas_cat['ycentroid'][sel], alpha=0.5, s=5, zorder=2)
        # sel=(meas_cat['kron_area_unmasked'] / meas_cat['kron_area']) > 0.9
        # tax.scatter(meas_cat['xcentroid'][sel], meas_cat['ycentroid'][sel], alpha=0.5, s=5)
        
        good = det_cat_full['good'] & \
            standard_meas_qual(meas_cat, min_snr=min_meas_snr) & \
            (meas_cat['y_shifts'] > min_yshifts) & \
            (meas_cat['y_shifts'] < max_yshifts) & \
            (np.isfinite(meas_cat['y_shifts'])) & \
            (meas_cat['background_total'] >= min_meas_background) & \
            (meas_cat['background_total'] <= max_meas_background)
#            (np.isfinite(meas_cat[xvar_name])) & \

        # limit x var if desired
        if min_xvar is not None:
            good &= (xvar >= min_xvar)
        if max_xvar is not None:
            good &= (xvar <= max_xvar)


        print(f'min/max xvar: {np.min(xvar[good])} - {np.max(xvar[good])}')


            #((meas_cat['kron_radius']*meas_cat['semimajor_sigma']) > 2) & \
            #(meas_cat['y_shifts'] > min_yshifts) & \
            #(meas_cat['y_shifts'] < max_yshifts) & \
            #(np.isfinite(meas_cat['y_shifts'])) & \
            #(snr_meas > min_meas_snr) & \
            #(np.isfinite(xvar)) & \
            #(meas_cat['MAG'] <= max_mag) & \
            #(meas_cat['MAG'] >= min_mag) & \
            #((meas_cat['area']/meas_cat['segment_area']) > 0.99) & \
            #((meas_cat['kron_area_unmasked'] / meas_cat['kron_area']) > 0.99) 

        # (det_cat['CLASS_STAR'] < 0.05) & (sources['FLAGS'] == 0)  & (det_cat['FLAGS'] <= 2) & (det_cat['KRON_RADIUS'] > 3.5)  & (det_cat['KRON_RADIUS'] < max_kron_rad)
        
        if use_det_cvar:
            cvar = det_cat_full[cvar_name]
        else:
            cvar = meas_cat[cvar_name]
        print(f'min/max cvar: {np.min(cvar[good])} - {np.max(cvar[good])}')

        row['prop_id'] = [prop_id]
        row['image_type'] = [image_type]
        row['cvarmin'] = [cvar_binmin]
        row['cvarmax'] = [cvar_binmax]

        ax = axes[i]
        sel = good
        if verbose:
            print(f'color variable range: {np.min(cvar[sel])} - {np.max(cvar[sel])}')
        # get some statistics on the scatter:
        sc_mean, sc_median, sc_std = sigma_clipped_stats(dmag[sel])
        #sc_mean_err = sc_std / np.sqrt(np.sum(sel))
        sc_mean_err = np.std(bootstrap(bootstrap(dmag[sel], bootfunc=np.nanmean, bootnum=1000)))
        #print(f'\nTotal scatter mean: {sc_mean}+/-{sc_mean_err}, median: {sc_median}, std: {sc_std}')
        row['mean'] = [sc_mean]
        row['mean_err'] = [sc_mean_err]
        #row['median'] = [sc_median]
        #row['std'] = [sc_std]

        spearman_res = spearmanr(xvar[sel], dmag[sel])
        #print(f'Total spearman correlation: {spearman_res.correlation}, p-value: {spearman_res.pvalue}')
        row['spearman_r'] = [spearman_res.correlation]
        row['spearman_p'] = [spearman_res.pvalue]
        # get theil-sen slope
        theilsen_results = theilslopes(dmag[sel], xvar[sel], alpha=0.68)
        #print(f'Theil-Sen slope: {theilsen_results[0]}, intercept: {theilsen_results[1]}')
        row['theil_slope'] = [theilsen_results[0]*1000]
        row['theil_slope_err'] = (theilsen_results[3] - theilsen_results[2]) / 2 * 1000
        #row['theil_intercept'] = [theilsen_results[1]]
        # row['theil_low_slope'] = [theilsen_results[2]]
        # row['theil_high_slope'] = [theilsen_results[3]]

        if first_time:
            tbl=row.copy()
            first_time = False
        else:
            tbl = vstack([tbl, row.copy()])

        if not means_only:
            pl=ax.scatter(xvar[sel], dmag[sel], s=2, c=meas_cat[cvar_name][sel], cmap=cvar_cmap, alpha=scatter_alpha, vmin=cvar_binmin, vmax=cvar_binmax)
            plt.colorbar(pl, label=cvar_label)

        #print(binmin,binmax, binsize)
        if xlim is None:
            ax.set_xlim(binmin-binsize/2,binmax+binsize/2)
        else:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.axhline(0,lw=3, ls='--', color='red', alpha=0.5)

        if prop_id == '13603': 
            ax.text(0.01,0.99,'2013',ha='left', va='top', transform=ax.transAxes, fontsize='x-large')
            #ax.set_title('2013')
        else:
            #ax.set_title('2021')
            ax.text(0.01,0.99,'2021',ha='left', va='top', transform=ax.transAxes, fontsize='x-large')

        if ylabel is None:
            ylabel=r'$\Delta mag$'
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # set up the secondary bins
        cvar_pc = np.percentile(cvar[np.isfinite(cvar)],[1,99])
        if cvar_binmin is None:
            cvar_binmin = cvar_pc[0] #np.nanmin(meas_cat[cvar_name])
        if cvar_binmax is None:
            cvar_binmax = cvar_pc[1] #np.nanmax(meas_cat[cvar_name])

        cvar_bins = np.linspace(cvar_binmin,cvar_binmax,cvar_nbins+1)
        cvar_binsmin = cvar_bins[:-1]
        cvar_binsmax = cvar_bins[1:]

        plt.set_cmap('Set2')

        bin_xshift = np.linspace(-1,1,len(cvar_binsmin))*binsize*bin_shift_mag
        if color_set is None:
            color_set = ['#e41a1c','#377eb8', '#4daf4a', '#984ea3']
        marker_set = ['o','v','P', 'D']
                
        for cmin, cmax, color, shift, marker in zip(cvar_binsmin, cvar_binsmax, color_set, bin_xshift, marker_set):
            print('\ncvar range: ', cmin, cmax)
            bin_mean = []
            bin_median = []
            bin_mean_err = []
            bin_mean_err_direct = []
            bin_std = []
            bin_n = []

            # get some stats of all the unbinned data first
            sel = good & (cvar >= cmin) & (cvar < cmax) & (np.isfinite(dmag))
            sc_mean, sc_median, sc_std = sigma_clipped_stats(dmag[sel])
            sc_mean_err = np.std(bootstrap(bootstrap(dmag[sel], bootfunc=np.nanmean, bootnum=1000)))
            #print(f'Scatter mean: {sc_mean}+/-{sc_mean_err}, median: {sc_median}, std: {sc_std}')
            spearman_res = spearmanr(xvar[sel], dmag[sel])
            #print(f'Spearman correlation: {spearman_res.correlation}, p-value: {spearman_res.pvalue}')

            theilsen_results = theilslopes(dmag[sel], xvar[sel], alpha=0.68)
            #print(f'Theil-Sen slope: {theilsen_results[0]}, intercept: {theilsen_results[1]}')


            row['cvarmin'] = [cmin]
            row['cvarmax'] = [cmax]
            row['mean'] = [sc_mean]
            row['mean_err'] = [sc_mean_err]
            row['spearman_r'] = [spearman_res.correlation]
            row['spearman_p'] = [spearman_res.pvalue]
            row['theil_slope'] = [theilsen_results[0]*1000]
            row['theil_slope_err'] = (theilsen_results[3] - theilsen_results[2]) / 2 * 1000
            #row['theil_intercept'] = [theilsen_results[1]]
            # row['theil_low_slope'] = [theilsen_results[2]]
            # row['theil_high_slope'] = [theilsen_results[3]]


            tbl = vstack([tbl, row.copy()])

            for lo, hi in zip(binsmin, binsmax):
                sel=good & (xvar >= lo) & (xvar < hi) & (cvar >= cmin) & (cvar < cmax) & (np.isfinite(dmag))
                mean, median, std = sigma_clipped_stats(dmag[sel])
                boots = bootstrap(dmag[sel], bootfunc=np.nanmedian, bootnum=100)
                bin_n.append(np.sum(sel))
                bin_median.append(median)
                bin_mean.append(mean)
                bin_std.append(std) 
                bin_mean_err.append(np.std(boots))
                bin_mean_err_direct.append(1/np.sum(sel)*np.sqrt(np.sum(dmag_err[sel]**2)))



            # bin_n = np.array(bin_n)))
            bin_n = np.array(bin_n)
            bin_mean = np.array(bin_mean)
            bin_std = np.array(bin_std)
            bin_mean_err = np.array(bin_mean_err)
            print('number per bin: ', bin_n)
            print('bin mean: ', bin_mean)
            print('bin mean_err: ', bin_mean_err_direct)

            ax.errorbar((binsmin + binsmax)/2 + shift, bin_median, bin_mean_err, marker=marker, ls='none', alpha=mean_alpha, color=color, label='{} - {}'.format(cmin, cmax), markersize=8,lw=3)

        if (i==0) & (not nolegend):
            ax.legend(title=cvar_label, loc=legend_loc)
    plot_title = 'Image Type = {}; Flux Type = {}'.format(image_type.upper(), flux_var)
    if title_suffix is not None:
        plot_title = plot_title + ' ;'+title_suffix
    plt.suptitle(plot_title, fontsize='x-large')



    if savefig is not None:
        plt.tight_layout()
        plt.savefig(savefig, dpi=150)
        
    return axes, tbl

def drc_readout_vector(drc_image, ref_flc, xcentroid, ycentroid, chip):
    # need to determine the readout direction and draw an arrow
    # define two positions with the same x_det and then use them to define a line
    #(this approach may not be general enough to work, but I think should work for
    # images with 90 degrees rotation like I have)
    #if (j == 0) & (i==0):
    
    size=100
    
    # load image
    image = fits.getdata(drc_image)
    
    x0 = int(xcentroid - size/2)
    xarr0 = np.ones(image.shape[1])*x0
    yarr0 = np.arange(image.shape[1])
    
    x1 = int(xcentroid + size/2)
    xarr1 = np.ones(image.shape[1])*x1
    yarr1 = np.copy(yarr0)
    
    if chip == 1:
        ext = 4
    else:
        ext = 1
        
    xarr_det0, yarr_det0 = pixtopix.tran(drc_image, ref_flc+f'[{ext}]', 'forward', x=xarr0, y=yarr0, verbose=False)
    xarr_det1, yarr_det1 = pixtopix.tran(drc_image, ref_flc+f'[{ext}]', 'forward', x=xarr1, y=yarr1, verbose=False)

    # get starting x position on detector
    middle = int(len(xarr_det0)/2)
    xdet0 = xarr_det0[middle]
    # corresponding image positions
    x_image0 = xarr0[middle]
    y_image0 = yarr0[middle]

    det_dx = np.abs(np.array(xarr_det1) - xdet0)
    ind_xdet1 = np.argmin(det_dx)
    # ind_xdet1 shows where x_detector is the same. Save corresponding points in the original image
    x_image1 = xarr1[ind_xdet1]
    y_image1 = yarr1[ind_xdet1]
    
    # define slope of line representing readout vector
    slope = (y_image1 - y_image0) / (x_image1 - x_image0)
    
    return slope

def mask_ro_side(image, x0, y0, chip, ro_vector_slope, plot=False, use_pa_axis=False, pa_slope=None):
    
    # define slope/intercept for readout direction vector passing through centroid
    if not use_pa_axis:
        ro_vector_int = y0 - ro_vector_slope * x0
        
        # define line perpendidular to this
        inv_slope = -1/ro_vector_slope
        inv_int = y0 - inv_slope * x0
    
    else:
        inv_slope = pa_slope
        inv_int = y0 - (pa_slope * x0)
        
    
    x_image_arr, y_image_arr = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    if not use_pa_axis:
        if chip == 1:
            mask = (y_image_arr > (inv_slope * x_image_arr + inv_int))
        else:
            mask = (y_image_arr < (inv_slope * x_image_arr + inv_int))
    else:
        if chip == 1:
            mask = (x_image_arr < ((y_image_arr - inv_int)/inv_slope))
        else:
            mask = (x_image_arr > ((y_image_arr - inv_int)/inv_slope))


    
    if plot:
        plotmask = np.copy(mask)
        plotmask[plotmask == 0] = np.nan

        fig, ax = plt.subplots()
        mean, med, std = sigma_clipped_stats(image)
        ax.imshow(image, origin='lower', aspect='auto', vmin=med - 3*std, vmax=med + 3*std)
        size=200
        xl = x0 - size/2
        xh = x0 + size/2
        yl = y0 - size/2
        yh = y0 + size/2
        
        ax.set_xlim(xl, xh)
        ax.set_ylim(yl, yh)
        
        # plot readout vector
        xvals = np.array([xl, xh])
        ax.plot(xvals, ro_vector_slope * xvals + ro_vector_int)
        
        # plot the perpendicular vector
        ax.plot(xvals, inv_slope*xvals + inv_int) 
        
        #fig, ax = plt.subplots()
        ax.imshow(plotmask, origin='lower', aspect='auto', cmap='Greys', alpha=0.75, interpolation='nearest')
        
    return mask

def all_cutouts(det_catalog, id, size=100, image_type='drc', 
                gauss_kernel_size = 9, gauss_kernel_fwhm = 4,
                show_segment=False, show_ratio=True):
    
    
    # define titles:
    titles = {}
    titles[13603] = {'JCH001010': '2013 - flash=0s',
                       'JCH001020': '2013 - flash=0.7s',
                       'JCH001030': '2013 - flash=1.4s'}
    titles[16870] = {'JERU01010': '2021 - flash=0s',
                       'JERU01020': '2021 - flash=0.7s',
                       'JERU01030': '2021 - flash=1.4s'}
    
    # set up the smoothing kernel for the ratio plot
    if gauss_kernel_fwhm > 0:
        smoothing_kernel = make_2dgaussian_kernel(fwhm=gauss_kernel_fwhm,
                                                  size=gauss_kernel_size)
    
    # load the detection catalog
    det_cat = Table.read(det_catalog)

    sel = det_cat['label'] == id
    source = det_cat[sel]

    print(source['label', 'xcentroid', 'ycentroid'])

    # holds the main cutouts
    fig, axes = plt.subplots(2,4, figsize=(15,8))
    
    if show_ratio:
        fig2, axes2 = plt.subplots(2,3,figsize=(12,8))

    # load the detection image
    det_file = '/Users/dstark/acs_work/cte/extended/data/processed/10325/10325_J91JC4010_drc_sci.fits'
    det_background_file = det_file.replace('.fits', '_photutils_background.fits')
    det_image = fits.getdata(det_file)
    det_image_background = fits.getdata(det_background_file)
    # subtract background
    det_image = det_image - det_image_background
    
    # convolve if desired
    if gauss_kernel_fwhm > 0:
        det_image = convolve(det_image, smoothing_kernel)

    
    exps = create_exposure_dictionary()
    sel = (exps['prop_id'] == '10325') & (exps['image_type'] == 'flc')
    ref_flc = exps['image_path'][sel][0]
    
    if show_segment:
        seg_file = '/Users/dstark/acs_work/cte/extended/data/processed/10325/10325_J91JC4010_drc_sci_photutils_segmentation_map.fits'
        segment = fits.getdata(seg_file)
        segment = segment.astype(float)
        segment[segment != id] = np.nan
        print('total in segment: ', np.nansum(np.isfinite(segment)))

    # get mag in det image
    det_photflam = fits.getval(det_file, 'PHOTFLAM')
    pxscl = np.abs(fits.getval(det_file,'CD1_1')*3600)  #arcsec/pixel
    source['segment_mag'] = -2.5*np.log10(source['segment_flux']*det_photflam) - 21.10
    source['kron_mag'] = -2.5*np.log10(source['kron_flux']*det_photflam) - 21.10

    dmean, dmedian, dstd = sigma_clipped_stats(det_image)
    cutout_truth = Cutout2D(det_image, (source['xcentroid'], source['ycentroid']), size)
    
    cutout_truth_arcsec2 = cutout_truth.data / pxscl**2
    cutout_truth_mag_arcsec2 = -2.5*np.log10((cutout_truth_arcsec2+100)*det_photflam)-21.1
    
    ax = axes[0,0]

    ax.imshow(cutout_truth.data, origin='lower', aspect='auto', vmin=-dstd, vmax=5*dstd, cmap='Greys')
    ax.set_title('2004 image')

    b_a = 1/source['elongation'][0] #np.sqrt(1-source['ellipticity'][0]**2)
    aperture = EllipticalAperture((size/2-0.5, size/2-0.5), source['kron_radius'][0]*2.*source['semimajor_sigma'][0], source['kron_radius'][0]*2.*source['semimajor_sigma'][0]*b_a, np.radians(source['orientation'][0]))
    aperture.plot(ax=ax, color='red',lw=2)

    if show_segment:
        seg_cutout = Cutout2D(segment, (source['xcentroid'], source['ycentroid']), size)
        ax.imshow(seg_cutout.data, origin='lower', aspect='auto', alpha=0.3, cmap='jet')

    if size is None:
        size = 1.25*2*source['kron_radius'][0]*2.*source['semimajor_sigma'][0]


    for prop_id, j in zip([13603, 16870], [0,1]):

        if prop_id == 13603:
            asns = ['JCH001010', 'JCH001020', 'JCH001030']
        elif prop_id == 16870:
            asns = ['JERU01010', 'JERU01020', 'JERU01030']
        for i, asn in zip([1,2,3], asns):
            meas_file = '/Users/dstark/acs_work/cte/extended/data/processed/{}/{}_{}_{}_sci_align.fits'.format(prop_id, prop_id, asn, image_type)
            meas_background_file = meas_file.replace('.fits','_photutils_background.fits')
            meas_photflam = fits.getval(meas_file,'PHOTFLAM')
            image = fits.getdata(meas_file)
            background = fits.getdata(meas_background_file)
            image = image - background
            
            if gauss_kernel_fwhm > 0:
                image = convolve(image, smoothing_kernel)
            
            
            mean, median, std = sigma_clipped_stats(image)
            
            meas_cat = '/Users/dstark/acs_work/cte/extended/data/processed/{}/{}_{}_{}_sci_align_photutils_cat.ecsv'.format(prop_id, prop_id, asn, image_type)
            sources_meas = Table.read(meas_cat)
            source_meas = sources_meas[sources_meas['label'] == id]
            
            cutout = Cutout2D(image, (source['xcentroid'], source['ycentroid']), size)
            cutout_arcsec2 = cutout.data / pxscl**2
            cutout_mag_arcsec2 = -2.5*np.log10((cutout_arcsec2+100)*det_photflam)-21.1
            
            
            # need to determine the readout direction and draw an arrow
            # define two positions with the same x_det and then use them to define a line
            #(this approach may not be general enough to work, but I think should work for
            # images with 90 degrees rotation like I have)
            #if (j == 0) & (i==0):
            x0 = int(source['xcentroid']-size/2)
            xarr0 = np.ones(image.shape[1])*x0
            yarr0 = np.arange(image.shape[1])
            chip = source_meas['chip']
            
            x1 = int(source['xcentroid']+size/2)
            xarr1 = np.ones(image.shape[1])*x1
            yarr1 = np.copy(yarr0)
            
            if chip == 1:
                ext = 4
            else:
                ext = 1
                
            xarr_det0, yarr_det0 = pixtopix.tran(det_file, ref_flc+f'[{ext}]', 'forward', x=xarr0, y=yarr0, verbose=False)
            xarr_det1, yarr_det1 = pixtopix.tran(det_file, ref_flc+f'[{ext}]', 'forward', x=xarr1, y=yarr1, verbose=False)

            # get starting x position on detector
            middle = int(len(xarr_det0)/2)
            xdet0 = xarr_det0[middle]
            # corresponding image positions
            x_image0 = xarr0[middle]
            y_image0 = yarr0[middle]

            det_dx = np.abs(np.array(xarr_det1) - xdet0)
            ind_xdet1 = np.argmin(det_dx)
            #print(xdet0, ind_xdet1, xarr_det1[ind_xdet1])
            # ind_xdet1 shows where x_detector is the same. Save corresponding points in the original image
            x_image1 = xarr1[ind_xdet1]
            y_image1 = yarr1[ind_xdet1]
            
            slope = (y_image1 - y_image0) / (x_image1 - x_image0)
            
            # draw these on the plot below

            ax = axes[j,i]
            ax.imshow(cutout.data, origin='lower', vmin=-dstd, vmax=10*dstd, cmap='Greys')
            ax.set_title(titles[prop_id][asn])
     #           '{}-{}-{}'.format(prop_id, asn, image_type))
            aperture.plot(ax=ax, color='red',lw=2)
            

            # load the catalog as well
            sources_meas = Table.read('/Users/dstark/acs_work/cte/extended/data/processed/{}/{}_{}_{}_sci_align_photutils_cat.ecsv'.format(prop_id, prop_id, asn, image_type))
            source_meas = sources_meas[sources_meas['label'] == id]
            source_meas['segment_mag'] = -2.5*np.log10(source_meas['segment_flux']*meas_photflam) - 21.10
            source_meas['kron_mag'] = -2.5*np.log10(source_meas['segment_flux']*meas_photflam) - 21.10
            background_total = source_meas['background_total'][0]


            dmag_iso = source['segment_mag'].value - source_meas['segment_mag'].value
            dmag_auto = source['kron_mag'].value - source_meas['kron_mag'].value

            ax.text(0.99,0.15,r'$\Delta$m(segment)={:.3f}'.format(dmag_iso[0]), transform=ax.transAxes, va='bottom',ha='right',bbox=dict(facecolor='white', alpha=0.75))
            ax.text(0.99,0.05,r'$\Delta$m(kron)={:.3f}'.format(dmag_auto[0]), transform=ax.transAxes, va='bottom',ha='right',bbox=dict(facecolor='white', alpha=0.75))

            # get the ratio image
            sel = (cutout.data > 1.5*std) & (cutout_truth.data > 1.5*dstd)    
            dmag = cutout_truth_mag_arcsec2 - cutout_mag_arcsec2
            dmag[~sel] = np.nan
            
            # if gauss_kernel_fwhm > 0:
            #     cutout_data_conv = convolve(cutout.data, smoothing_kernel)
            #     cutout_truth_data_conv = convolve(cutout_truth.data, smoothing_kernel)
            #     ratio_image = cutout_data_conv/cutout_truth_data_conv
            #     sel = (cutout.data > 1*dstd) & (cutout_truth.data > 1*std)    
            #     ratio_image[~sel] = np.nan
            # else:
            #     sel = (cutout.data > 2*dstd) & (cutout_truth.data > 2*std)    
            #     ratio_image = cutout.data / cutout_truth.data
            #     ratio_image[~sel] = np.nan
            ax2 = axes2[j,i-1]
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            
            #ax2.set_title(titles[prop_id][asn])
            ax2.text(0.99, 0.99, titles[prop_id][asn].split('-')[0] + f'- bkg={background_total:.1f}', ha='right', va='top', transform=ax2.transAxes, fontsize='x-large')

            #p=ax2.imshow(ratio_image, origin='lower', aspect='auto', vmin=0.5, vmax=1.5, cmap='coolwarm')
            p=ax2.imshow(dmag, origin='lower', aspect='auto', vmin=-0.05, vmax=.05, cmap='coolwarm')
            if (j==1) & (i==2):
                cax = ax2.inset_axes([0., 1.17, 1, .05])
                fig.colorbar(p, cax=cax, orientation='horizontal', label=r'$\Delta$ mag')

            ro_endpoints = np.array([0.25, 0.75])
            
            if chip == 1:
                xy = (0.25*size, 0.25*size)
                xytext = (0.75*size, 0.25*size + slope*(0.75-0.25)*size)
            else:
                xytext = (0.25*size, 0.25*size)
                xy = (0.75*size, 0.25*size + slope*(0.75-0.25)*size)
            ax2.annotate("", xytext=xytext, xy=xy, arrowprops=dict(arrowstyle="->", lw=3))
            
     
       
    #cax = ax.inset_axes([0.3, 0.07, 0.4, 0.04])
    #plt.colorbar(p)
    plt.subplots_adjust(hspace=0.25)

    axes[1,0].axis("off")

def ratio_cutouts(det_catalog, id, comp_prop_id, size=100, image_type='drc', 
                gauss_kernel_size = 9, gauss_kernel_fwhm = 4,
                show_segment=False):
    
    
    # define titles:
    titles = {}
    titles[13603] = {'JCH001010': '2013 - flash=0s',
                       'JCH001020': '2013 - flash=0.7s',
                       'JCH001030': '2013 - flash=1.4s'}
    titles[16870] = {'JERU01010': '2021 - flash=0s',
                       'JERU01020': '2021 - flash=0.7s',
                       'JERU01030': '2021 - flash=1.4s'}
    
    # set up the smoothing kernel for the ratio plot
    if gauss_kernel_fwhm > 0:
        smoothing_kernel = make_2dgaussian_kernel(fwhm=gauss_kernel_fwhm,
                                                  size=gauss_kernel_size)
    
    # load the detection catalog
    det_cat = Table.read(det_catalog)

    sel = det_cat['label'] == id
    source = det_cat[sel]

    # set a default cutout size if needed
    if size is None:
        size = 1.25*2*source['kron_radius'][0]*2.*source['semimajor_sigma'][0]

    print(source['label', 'xcentroid', 'ycentroid'])

    # holds the main cutouts
    fig, axes = plt.subplots(2,4, figsize=(15,8))
    # top row is detection image + measurement images at different backgrounnd (flash) levels
    # bottom row is the ratio images (first column empty w/ axes off)
    
    # load the detection image
    det_file = '/Users/dstark/acs_work/cte/extended/data/processed/10325/10325_J91JC4010_drc_sci.fits'
    det_background_file = det_file.replace('.fits', '_photutils_background.fits')
    det_image = fits.getdata(det_file)
    det_image_background = fits.getdata(det_background_file)
    # subtract background
    det_image = det_image - det_image_background
    
    # convolve if desired
    if gauss_kernel_fwhm > 0:
        det_image = convolve(det_image, smoothing_kernel)

    exps = create_exposure_dictionary()
    sel = (exps['prop_id'] == '10325') & (exps['image_type'] == 'flc')
    ref_flc = exps['image_path'][sel][0]
    
    # collect the segmentation image if desired
    if show_segment:
        seg_file = '/Users/dstark/acs_work/cte/extended/data/processed/10325/10325_J91JC4010_drc_sci_photutils_segmentation_map.fits'
        segment = fits.getdata(seg_file)
        segment = segment.astype(float)
        segment[segment != id] = np.nan
        #print('total in segment: ', np.nansum(np.isfinite(segment)))

    # get magnitude in det image
    det_photflam = fits.getval(det_file, 'PHOTFLAM')
    source['segment_mag'] = -2.5*np.log10(source['segment_flux']*det_photflam) - 21.10
    source['kron_mag'] = -2.5*np.log10(source['kron_flux']*det_photflam) - 21.10

    # statistics on the detection image
    dmean, dmedian, dstd = sigma_clipped_stats(det_image)

    # cutout of detection image
    cutout_truth = Cutout2D(det_image, (source['xcentroid'], source['ycentroid']), size)
    
    # convert pixel values into mag/arcsec. Bias level of 100 added to avoid things going negative
    pxscl = np.abs(fits.getval(det_file,'CD1_1')*3600)  #arcsec/pixel
    cutout_truth_arcsec2 = cutout_truth.data / pxscl**2
    cutout_truth_mag_arcsec2 = -2.5*np.log10((cutout_truth_arcsec2+100)*det_photflam)-21.1
    
    # plot the detection image cutout
    ax = axes[0,0]
    ax.imshow(cutout_truth.data, origin='lower', aspect='auto', vmin=-dstd, vmax=5*dstd, cmap='Greys')
    ax.set_title('2004 image')

    # overlay the standard Kron aperture
    b_a = 1/source['elongation'][0]
    aperture = EllipticalAperture((size/2-0.5, size/2-0.5), 
                                    source['kron_radius'][0]*2.*source['semimajor_sigma'][0], 
                                    source['kron_radius'][0]*2.*source['semimajor_sigma'][0]*b_a, 
                                    np.radians(source['orientation'][0]))
    aperture.plot(ax=ax, color='red',lw=2)

    # overlay segmentation image if desired
    if show_segment:
        seg_cutout = Cutout2D(segment, (source['xcentroid'], source['ycentroid']), size)
        ax.imshow(seg_cutout.data, origin='lower', aspect='auto', alpha=0.3, cmap='jet')

    # Next, cycle through the measurement images from whatever prop-id is set

    # determine the ASNs for the different flash levels
    ### to do: automate the selection of the asns
    if comp_prop_id == 13603:
        asns = ['JCH001010', 'JCH001020', 'JCH001030']
    elif comp_prop_id == 16870:
        asns = ['JERU01010', 'JERU01020', 'JERU01030']

    for i, asn in zip([1,2,3], asns):
        # load the measurement image, subtract background
        meas_file = '/Users/dstark/acs_work/cte/extended/data/processed/{}/{}_{}_{}_sci_align.fits'.format(comp_prop_id, comp_prop_id, asn, image_type)
        meas_background_file = meas_file.replace('.fits','_photutils_background.fits')
        meas_photflam = fits.getval(meas_file,'PHOTFLAM')
        image = fits.getdata(meas_file)
        background = fits.getdata(meas_background_file)
        image = image - background
        
        # convolve measurement image if desired
        if gauss_kernel_fwhm > 0:
            image = convolve(image, smoothing_kernel)
        
        # get some stats on measurement image
        mean, median, std = sigma_clipped_stats(image)
        
        # load the corresponding measurement image catalog. Will need this later??
        meas_cat = '/Users/dstark/acs_work/cte/extended/data/processed/{}/{}_{}_{}_sci_align_photutils_cat.ecsv'.format(comp_prop_id, comp_prop_id, asn, image_type)
        sources_meas = Table.read(meas_cat)
        source_meas = sources_meas[sources_meas['label'] == id]
        source_meas['segment_mag'] = -2.5*np.log10(source_meas['segment_flux']*meas_photflam) - 21.10
        source_meas['kron_mag'] = -2.5*np.log10(source_meas['segment_flux']*meas_photflam) - 21.10
        background_total = source_meas['background_total'][0]

        # create cutout of measurement image object; convert cutout to mag/arcsec^2
        cutout = Cutout2D(image, (source['xcentroid'], source['ycentroid']), size)
        cutout_arcsec2 = cutout.data / pxscl**2
        cutout_mag_arcsec2 = -2.5*np.log10((cutout_arcsec2+100)*det_photflam)-21.1
        
        
        # need to determine the readout direction and draw an arrow
        # define two positions with the same x_det and then use them to define a line
        #(this approach may not be general enough to work, but I think should work for
        # images with 90 degrees rotation like I have)
        #if (j == 0) & (i==0):

        slope = drc_readout_vector(det_file, ref_flc, source['xcentroid'], source['ycentroid'], source_meas['chip'])

        # x0 = int(source['xcentroid']-size/2)
        # xarr0 = np.ones(image.shape[1])*x0
        # yarr0 = np.arange(image.shape[1])
        # chip = source_meas['chip']
        
        # x1 = int(source['xcentroid']+size/2)
        # xarr1 = np.ones(image.shape[1])*x1
        # yarr1 = np.copy(yarr0)
        
        # if chip == 1:
        #     ext = 4
        # else:
        #     ext = 1
            
        # xarr_det0, yarr_det0 = pixtopix.tran(det_file, ref_flc+f'[{ext}]', 'forward', x=xarr0, y=yarr0, verbose=False)
        # xarr_det1, yarr_det1 = pixtopix.tran(det_file, ref_flc+f'[{ext}]', 'forward', x=xarr1, y=yarr1, verbose=False)

        # # get starting x position on detector
        # middle = int(len(xarr_det0)/2)
        # xdet0 = xarr_det0[middle]
        # # corresponding image positions
        # x_image0 = xarr0[middle]
        # y_image0 = yarr0[middle]

        # det_dx = np.abs(np.array(xarr_det1) - xdet0)
        # ind_xdet1 = np.argmin(det_dx)
        # #print(xdet0, ind_xdet1, xarr_det1[ind_xdet1])
        # # ind_xdet1 shows where x_detector is the same. Save corresponding points in the original image
        # x_image1 = xarr1[ind_xdet1]
        # y_image1 = yarr1[ind_xdet1]
        
        # slope = (y_image1 - y_image0) / (x_image1 - x_image0)
        
        # draw these on the plot below

        # plot the measurement image cutout and the standard kron aperture
        ax = axes[0,i]
        ax.imshow(cutout.data, origin='lower', vmin=-dstd, vmax=10*dstd, cmap='Greys')
        ax.set_title(titles[comp_prop_id][asn].split('-')[-1] + f' - bkg={background_total:.1f} e$^-$/pix')
    #           '{}-{}-{}'.format(prop_id, asn, image_type))
        aperture.plot(ax=ax, color='red',lw=2)
        
        # calculate delta_mag in case I want to display this
        dmag_iso = source['segment_mag'].value - source_meas['segment_mag'].value
        dmag_auto = source['kron_mag'].value - source_meas['kron_mag'].value

        # ax.text(0.99,0.15,r'$\Delta$m(segment)={:.3f}'.format(dmag_iso[0]), transform=ax.transAxes, va='bottom',ha='right',bbox=dict(facecolor='white', alpha=0.75))
        # ax.text(0.99,0.05,r'$\Delta$m(kron)={:.3f}'.format(dmag_auto[0]), transform=ax.transAxes, va='bottom',ha='right',bbox=dict(facecolor='white', alpha=0.75))

        # get the ratio image; mask regions with especially low SNR
        dmag = cutout_mag_arcsec2 - cutout_truth_mag_arcsec2 
        sel = (cutout.data > 2*std) & (cutout_truth.data > 2*dstd)    
        dmag[~sel] = np.nan
        
        # if gauss_kernel_fwhm > 0:
        #     cutout_data_conv = convolve(cutout.data, smoothing_kernel)
        #     cutout_truth_data_conv = convolve(cutout_truth.data, smoothing_kernel)
        #     ratio_image = cutout_data_conv/cutout_truth_data_conv
        #     sel = (cutout.data > 1*dstd) & (cutout_truth.data > 1*std)    
        #     ratio_image[~sel] = np.nan
        # else:
        #     sel = (cutout.data > 2*dstd) & (cutout_truth.data > 2*std)    
        #     ratio_image = cutout.data / cutout_truth.data
        #     ratio_image[~sel] = np.nan
        ax2 = axes[1,i]
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        
        #ax2.set_title(titles[prop_id][asn])
        #ax2.text(0.99, 0.99, titles[comp_prop_id][asn].split('-')[0] + f'- bkg={background_total:.1f}', ha='right', va='top', transform=ax2.transAxes, fontsize='x-large')

        # plotthe dmag image
        p=ax2.imshow(dmag, origin='lower', aspect='auto', vmin=-0.05, vmax=.05, cmap='coolwarm')
        if i==1:
            cax = ax2.inset_axes([-0.1, 0, 0.05, 1])
            fig.colorbar(p, cax=cax, label=r'$\Delta$ mag/arcsec^2', location='left')

        #overlay the readout vector now
        ro_endpoints = np.array([0.25, 0.75])
        
        if source_meas['chip'] == 1:
            xy = (0.25*size, 0.25*size)
            xytext = (0.75*size, 0.25*size + slope*(0.75-0.25)*size)
        else:
            xytext = (0.25*size, 0.25*size)
            xy = (0.75*size, 0.25*size + slope*(0.75-0.25)*size)
        ax2.annotate("", xytext=xytext, xy=xy, arrowprops=dict(arrowstyle="->", lw=3))
            
    #cax = ax.inset_axes([0.3, 0.07, 0.4, 0.04])
    #plt.colorbar(p)
    plt.subplots_adjust(hspace=0.25)

    axes[1,0].axis("off")

def custom_cutouts(data, positions, shapes, savedir, names, segmentation=None, background=None, segmentation_labels=None):
    
    # to do: check that segmentation and background have the same size
    save_path = Path(savedir)
    # make sure the path exists
    save_path.mkdir(exist_ok=True)
    print('saving to '+str(save_path))
    
    for position, shape, name, label in zip(positions, shapes, names, segmentation_labels):
        print(name)
        cutout_data = CutoutImage(data, position, shape)
        fits.writeto(Path.joinpath(save_path, Path(name + '_cutout.fits')), cutout_data.data, overwrite=True)
        print(str(Path.joinpath(save_path, name + '_cutout.fits')))
        
        if (segmentation is not None) & (segmentation_labels is not None):
            
            cutout_segmentation = CutoutImage((segmentation == label).astype(int), position, shape)
            fits.writeto(Path.joinpath(save_path, name + '_cutout_segmentation.fits'), cutout_segmentation.data, overwrite=True)

        if background is not None:
            
            cutout_background = CutoutImage(background, position, shape)
            fits.writeto(Path.joinpath(save_path, name + '_cutout_background.fits'), cutout_background.data, overwrite=True)

from photutils.aperture import aperture_photometry, EllipticalAnnulus
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats


def radial_profile(image, header, x0, y0, angle, radii, b_a, error_image = None, mask=None, plot=False, localbkg_width=32):
    
    # get pixel scale from header
    pxscl = np.abs(header['CD1_1']*3600)  #arcsec/pixel
    photflam = header['PHOTFLAM']
    
    # set up apertures
    apertures = [EllipticalAperture((x0, y0), radius, radius * b_a, angle) for radius in radii]
    
    # run aperture photometry
    ap_phot = aperture_photometry(image, apertures, error = error_image, mask=mask)
    
    # subtract a local background if desired
    if localbkg_width > 0:
        scale=1.5
        bkg_aperture = EllipticalAnnulus((x0, y0), scale*radii[-1], 
                                         scale*radii[-1]+localbkg_width,
                                         (scale*radii[-1]+localbkg_width)*b_a,
                                         theta=angle)
        bkg_phot = aperture_photometry(image, bkg_aperture, mask=mask)
        bkg_area = bkg_aperture.area_overlap(image, mask=mask)
        local_background = bkg_phot['aperture_sum'] / bkg_area # counts/s/pix
    else:
        local_background = 0.
        

    # initialize table that holds surface brightness
    surface_brightness = Table()
    
    # create a new table holding the SBprob
    for i, aperture in enumerate(apertures):
        if i == 0:
            area = aperture.area_overlap(image, mask=mask)
            area_arcsec2 = area * pxscl * pxscl
            flux = ap_phot[f'aperture_sum_{i}']
        else:
            area = aperture.area_overlap(image, mask=mask) - apertures[i-1].area_overlap(image, mask=mask)
            area_arcsec2 = area * pxscl * pxscl
            flux = ap_phot[f'aperture_sum_{i}'] - ap_phot[f'aperture_sum_{i-1}']
        
        # add areas to the tables
        surface_brightness[f'area_pix_{i}'] = [area]
        surface_brightness[f'area_arsec2_{i}'] = [area_arcsec2]
        
        # add surface brightness mesurements to the table now
        surface_brightness[f'aperture_sb_{i}'] = flux/area - local_background
        surface_brightness[f'aperture_sb_arcsec2_{i}'] = surface_brightness[f'aperture_sb_{i}']/pxscl**2   #flux / area_arcsec2
        surface_brightness[f'aperture_sb_mag_arcsec2_{i}'] = -2.5 * np.log10(surface_brightness[f'aperture_sb_arcsec2_{i}']*photflam) - 21.1
        
        # indicate radius info
        surface_brightness[f'radius_{i}'] = aperture.a
    
    if plot:
        fig, ax = plt.subplots()
        mean, median, std = sigma_clipped_stats(image)
        ax.imshow(image, origin='lower', vmin=median-std, vmax=median+5*std, cmap='viridis')
        ax.set_xlim(x0-np.max(radii), x0+np.max(radii))
        ax.set_ylim(y0-np.max(radii), y0+np.max(radii))
        for ap in apertures:
            ap.plot(ax)
            
        # # overlay mask
        #fig, ax = plt.subplots()
        plotmask = np.copy(mask)
        ax.imshow(plotmask, origin='lower', cmap='Reds', interpolation='nearest', alpha=0.75)
        
    
    return apertures, ap_phot, surface_brightness

# def surface_brightness_profile(apertures, ap_phot, photflam=None, wcs=None):
#     '''return counts/s/pix of photflam or wcs are empty. Otherwise return mag/arcsec'''
#     radii = 
from astropy.nddata import block_reduce

def combine_profiles(sbprof_files, measurement_cat_files, ref_sbprof_file, 
                     det_cat_file, min_det_snr=10, min_meas_snr=3,
                     max_yshifts = 9999, min_yshifts=0, max_mag = 26,
                     min_mag = 21.5, max_star_prob = 0.3,
                     cvar_name = 'background_total'
                     ):
    
    first_time = True
    for sbprof_file, measurement_cat_file in zip(sbprof_files, measurement_cat_files):
        
        # load the sb prof and the corresponding reference. These should be entirely matched
        sb = Table.read(sbprof_file)
        ref_sb = Table.read(ref_sbprof_file)
        
        # put a proper error code here
        if (len(sb) != len(ref_sb)) | (np.sum(sb['label'] != ref_sb['label']) > 0):
            print('SURFACE BRIGHTNESS TABLE DOESNT MATCH THE REFERENCE...QUITTING')
            return
                
        # load measurement and detection catalogs
        meas_cat = Table.read(measurement_cat_file)
        det_cat = Table.read(det_cat_file)
                
        # pull out flux cal for both meas and ref files. Add them to table so we can 
        # average things with different flux cals efectively
        meas_image = measurement_cat_file.split('_photutils')[0] + '.fits'
        meas_photflam = fits.getval(meas_image, 'PHOTFLAM')
        sb['photflam'] = meas_photflam
        
        det_image = det_cat_file.split('_photutils')[0] + '.fits'
        det_photflam = fits.getval(det_image, 'PHOTFLAM')
        ref_sb['photflam'] = det_photflam
        det_cat.meta['PHOTFLAM'] = det_photflam
        

        
    
        y_shifts = meas_cat['y_shifts']
        
        good_det = standard_det_qual(det_cat) & \
            ((det_cat['orientation'] < -36.3) | (det_cat['orientation'] > 23.7))
        #(det_snr > min_det_snr) & (star_prob < max_star_prob) & \
                    #det_goodregion & (det_mag < max_mag) & (det_mag > min_mag)
        
        good_meas = standard_meas_qual(meas_cat, min_snr=min_meas_snr, max_yshifts=2048, 
                       min_yshifts=0, min_segment_area=0.99, 
                       min_kron_area=0.99, flux_var='kron_flux')

        good_meas = good_meas & (y_shifts <= max_yshifts) & (y_shifts >= min_yshifts)
        
        # (meas_snr > min_meas_snr) & (y_shifts <= max_yshifts) & \
        #             (y_shifts >= min_yshifts) & (good_segment_area > 0.99) & \
        #             (good_kron_area > 0.99)
                    
        good_labels = det_cat['label'][good_det & good_meas]
        
        #subselect part of table corresponding to the good data
        keep = np.array([s['label'] in good_labels for s in sb])
        subsb = sb[keep]
        subsb[cvar_name] = -999.
        
        # do the same for the reference sbs
        ref_keep = np.array([s['label'] in good_labels for s in ref_sb])
        ref_subsb = ref_sb[ref_keep]
        
        # add the cvar now (just measurement table)
        for s in subsb:
            sel = np.where((meas_cat['label'] == s['label']))[0]
            s[cvar_name] = meas_cat[cvar_name][sel]
            
        if first_time:
            master_sb = subsb
            master_ref_sb = ref_subsb
            first_time = False
        else:
            master_sb = vstack([master_sb, subsb])
            master_ref_sb = vstack([master_ref_sb, ref_subsb])
            
    return master_sb, master_ref_sb

def bin_profiles(master_sb, subsel=None, min_area_pix=10, starting_ind=0):
    
    columns = master_sb.columns
    
    medval_arr = []
    medval_arr_err = []
    
    i=np.copy(starting_ind)
    proceed=True
    while proceed:
        sel = np.isfinite(master_sb[f'aperture_sb_arcsec2_{i}']) & (master_sb[f'area_pix_{i}'] > min_area_pix)
        if subsel is not None:
            sel = sel & subsel 
            
        print(f'total in radial bin {i}: ', np.sum(sel))
    
        # get median value
        mean, median, std = sigma_clipped_stats(master_sb[f'aperture_sb_arcsec2_{i}'][sel] * master_sb['photflam'][sel])
        medval = mean #median #np.mean(master_sb[f'aperture_sb_arcsec2_{i}'][sel] * master_sb['photflam'][sel])
        # convert to ST mag
        medval_mag = -2.5*np.log10(medval) - 21.1

        # get uncertainty via bootstrap
        samples = bootstrap(master_sb[f'aperture_sb_arcsec2_{i}'][sel] * master_sb['photflam'][sel], bootfunc=np.nanmean, bootnum=1000)
        samples_mag = -2.5*np.log10(samples) - 21.1
        medval_mag_err = np.nanstd(samples_mag)     
        
        # append derived values to lists
        medval_arr.append(medval_mag)
        medval_arr_err.append(medval_mag_err)
    
        # increase i and check that next column exists
        i += 1
        if f'aperture_sb_arcsec2_{i}' not in columns:
            proceed = False
    
    # convert to arrays anr return
    medval_arr = np.array(medval_arr)
    medval_arr_err = np.array(medval_arr_err)  

    return medval_arr, medval_arr_err


def sbprofs_background2(sbprof_files, ref_sbprof_file, det_cat_file, 
                       measurement_cat_files, cvar_binmin=10, cvar_binmax=40,
                       cvar_nbins=3, bin_shift_mag=0.05, rebin=0,
                       max_mag=26, min_mag=21.5, max_star_prob=0.3, min_yshifts = 0,
                       max_yshifts=2048, min_samples=10, dmag=True, min_det_snr=10,
                       min_meas_snr=10, cvar_name = 'background_total', ax=None,
                       ls='-', label_suffix='', add_xshift=0,
                       plotcolor=None,noylabel=False, nolegend=False, min_area_pix=10,
                       starting_ind=0, line_label=None):
    
    print('\n STARTING MERGING PROFILES\n')

    kron_frac = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])  # find better way to do this so the kron fracs are not hard coded!
    half_bin = kron_frac[1] - kron_frac[0]

    master_sb, master_ref_sb = combine_profiles(sbprof_files, 
                                                measurement_cat_files,
                                                ref_sbprof_file, 
                                                det_cat_file, 
                                                min_det_snr=min_det_snr, 
                                                min_meas_snr=min_meas_snr, 
                                                max_yshifts=max_yshifts, 
                                                min_yshifts=min_yshifts, 
                                                max_mag=max_mag, 
                                                min_mag=min_mag, 
                                                max_star_prob=max_star_prob,
                                                cvar_name=cvar_name)
    ## DONE FILTERING AND ACCUMULATING DATA. START PLOTTING

    # set up 3rd variable bins
    cvar_pc = np.percentile(master_sb[cvar_name][np.isfinite(master_sb[cvar_name])],[1,99])
    if cvar_binmin is None:
        cvar_binmin = cvar_pc[0] 
    if cvar_binmax is None:
        cvar_binmax = cvar_pc[1] 

    cvar_bins = np.linspace(cvar_binmin,cvar_binmax,cvar_nbins+1)
    cvar_binsmin = cvar_bins[:-1]
    cvar_binsmax = cvar_bins[1:]
    
    # initialize plot if ax is not set already
    if ax is None:
        fig, ax = plt.subplots()
        if not dmag:
            plt.gca().invert_yaxis()
        
    # median the reference sbprofs and plot
   
    #in_cvar_bin = (master_sb[cvar_name] >= cvar_binmin) & (master_sb[cvar_name] <= cvar_binmax)
    # ref_medval_arr, ref_medval_arr_err = bin_profiles(master_ref_sb, subsel=in_cvar_bin)
    
    # # rebin if needed (put this in its own function)
    # if rebin > 1:
    #     ref_medval_arr = block_reduce(ref_medval_arr, rebin, np.mean)
    #     ref_medval_arr_err = block_reduce(ref_medval_arr_err**2, rebin, np.sum)
    #     ref_medval_arr_err = np.sqrt(ref_medval_arr_err/rebin)
    #     kron_frac_plot = block_reduce(kron_frac, rebin, np.mean)
    # else:
    #     kron_frac_plot = kron_frac
    
    # if not dmag:
    
    #     ax.errorbar(kron_frac_plot, ref_medval_arr, ref_medval_arr_err, color='black',
    #                 ls=ls, label='truth' + label_suffix)
        
    
    # median the measurement sbprofs and plot
    color_set = ['#e41a1c','#377eb8', '#4daf4a', 'magenta']
    binsize = kron_frac[1] - kron_frac[0]
    bin_xshift = np.linspace(-1,1,len(cvar_binsmin))*binsize*bin_shift_mag

    for lo, hi, color, shift in zip(cvar_binsmin, cvar_binsmax, color_set, bin_xshift):

        if plotcolor is None:
            plotcolor=color

        in_bin = (master_sb[cvar_name] >= lo) & (master_sb[cvar_name] <= hi)

        # get values for reference sbprofs. Select on same range of backgrounds as later epoch data for consistency
        ref_medval_arr, ref_medval_arr_err = bin_profiles(master_ref_sb, subsel=in_bin, min_area_pix=min_area_pix, starting_ind=starting_ind)
        
        # rebin if needed (put this in its own function)
        if rebin > 1:
            ref_medval_arr = block_reduce(ref_medval_arr, rebin, np.mean)
            ref_medval_arr_err = block_reduce(ref_medval_arr_err**2, rebin, np.sum)
            ref_medval_arr_err = np.sqrt(ref_medval_arr_err/rebin)
            kron_frac_plot = block_reduce(kron_frac, rebin, np.mean)
        else:
            kron_frac_plot = kron_frac
        
        if not dmag:
        
            ax.errorbar(kron_frac_plot, ref_medval_arr, ref_medval_arr_err, color='black',
                        ls=ls, label='truth' + label_suffix)

        # same for measurement array
        medval_arr, medval_arr_err = bin_profiles(master_sb, subsel=in_bin, min_area_pix=min_area_pix, starting_ind=starting_ind)
        
        if rebin > 1:
            medval_arr = block_reduce(medval_arr, rebin, np.mean)
            medval_arr_err = block_reduce(medval_arr_err**2, rebin, np.sum)
            medval_arr_err = np.sqrt(medval_arr_err/rebin)        
        
        if dmag:
            sb_dmag = medval_arr - ref_medval_arr
            sb_dmag_err = np.sqrt(medval_arr_err**2 + ref_medval_arr_err**2)
            print('sb_dmag: ', sb_dmag)
            print('sb_dmag_err: ', sb_dmag_err)
            ax.errorbar(kron_frac_plot+shift+add_xshift - half_bin, sb_dmag, sb_dmag_err, 
            color=plotcolor,label=line_label, ls=ls)
            
        else:
            ax.errorbar(kron_frac_plot+shift+add_xshift - half_bin, medval_arr, medval_arr_err, color=plotcolor,
                        label='{} - {}'.format(lo, hi) + label_suffix, ls=ls)



    if not nolegend:
        ax.legend(title='background [e-/s]')
    ax.set_xlabel('r / r$_{kron}$')
    if not noylabel:
        if dmag:
            ax.set_ylabel('$\Delta$ mag arcsec$^{-2}$')
        else:
            ax.set_ylabel('mag arcsec$^{-2}$')
    if dmag:
        ax.axhline(0,lw=3,ls='--',alpha=0.7)
    
    print('/n DONE /n')
            
    return ax, sb_dmag, sb_dmag_err
            
        
    
def sbprofs_background(sbprof_files, ref_sbprof_file, det_cat_file, 
                       measurement_cat_files, cvar_binmin=10, cvar_binmax=40,
                       cvar_nbins=3, bin_shift_mag=0.1, rebin=0,
                       max_mag=26.5, max_star_prob=0.9, min_yshifts = 0,
                       max_yshifts=2048, min_samples=10, dmag=True, min_det_snr=10,
                       min_meas_snr=5,cvar_name = 'background_total'):
    
    kron_frac = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])  # find better way to do this so the kron fracs are not hard coded!

    
    # cycle through profile files
    first_time = True
    for sbprof_file, measurement_cat_file in zip(sbprof_files, measurement_cat_files):
        
        sb = Table.read(sbprof_file)
        ref_sb = Table.read(ref_sbprof_file)
        
        if len(sb) != len(ref_sb):
            print('SURFACE BRIGHTNESS TABLE DOESNT MATCH THE REFERENCE...QUITTING')
            return
                
        meas_cat = Table.read(measurement_cat_file)
        det_cat = Table.read(det_cat_file)
        
        print(len(sb), len(meas_cat), len(det_cat))
        
        # pull out flux cal
        meas_image = measurement_cat_file.split('_photutils')[0] + '.fits'
        meas_photflam = fits.getval(meas_image, 'PHOTFLAM')
        # put this in the table so we can average things with different flux cals effectively
        sb['photflam'] = meas_photflam
        
        det_image = det_cat_file.split('_photutils')[0] + '.fits'
        det_photflam = fits.getval(det_image, 'PHOTFLAM')
        ref_sb['photflam'] = det_photflam
        
        # get variables from det and meas cat needed for selections
        det_snr = det_cat['kron_flux']/det_cat['kron_fluxerr']
        star_prob = det_cat['star_prob']
        det_mag = -2.5*np.log10(det_cat['kron_flux'] * det_photflam) - 21.1
        det_goodregion = (det_cat['xcentroid'] < (det_cat['ycentroid']/27.51+6417)) & \
                         (det_cat['ycentroid'] > (-0.112*det_cat['xcentroid']+745.2)) & \
                         (det_cat['ycentroid'] < (-0.089*det_cat['xcentroid'] + 6741.1)) & \
                         (det_cat['xcentroid'] > (det_cat['ycentroid']/109.15+322.4))
    
        meas_snr = meas_cat['kron_flux'] / meas_cat['kron_fluxerr']
        y_shifts = meas_cat['y_shifts']
        good_segment_area = meas_cat['area']/meas_cat['segment_area']
        good_kron_area = meas_cat['kron_area_unmasked'] / meas_cat['kron_area']
        
        good_det = (det_snr > min_det_snr) & (star_prob < max_star_prob) & det_goodregion & (det_mag < max_mag)
        
        good_meas = (meas_snr > min_meas_snr) & (y_shifts <= max_yshifts) & \
                    (y_shifts >= min_yshifts) & (good_segment_area > 0.9) & \
                    (good_kron_area > 0.9)
                    
        good_labels = det_cat['label'][good_det & good_meas]
        
        #subselect part of table corresponding to the good data
        keep = np.array([s['label'] in good_labels for s in sb])
        subsb = sb[keep]
        subsb[cvar_name] = -999.
        
        # do the same for the reference sbs
        ref_keep = np.array([s['label'] in good_labels for s in ref_sb])
        ref_subsb = ref_sb[ref_keep]
        
        # add the cvar now (just measurement table)
        for s in subsb:
            sel = np.where((meas_cat['label'] == s['label']))[0]
            s[cvar_name] = meas_cat[cvar_name][sel]
            
        if first_time:
            master_sb = subsb
            master_ref_sb = ref_subsb
            first_time = False
        else:
            master_sb = vstack([master_sb, subsb])
            master_ref_sb = vstack([master_ref_sb, ref_subsb])
            
    ## DONE FILTERING AND ACCUMULATING DATA. START PLOTTING

    # set up the bins
    cvar_pc = np.percentile(master_sb[cvar_name][np.isfinite(master_sb[cvar_name])],[1,99])
    if cvar_binmin is None:
        cvar_binmin = cvar_pc[0] #np.nanmin(meas_cat[cvar_name])
    if cvar_binmax is None:
        cvar_binmax = cvar_pc[1] #np.nanmax(meas_cat[cvar_name])

    cvar_bins = np.linspace(cvar_binmin,cvar_binmax,cvar_nbins+1)
    cvar_binsmin = cvar_bins[:-1]
    cvar_binsmax = cvar_bins[1:]
    
    # initialize plot
    fig, ax = plt.subplots()
    plt.gca().invert_yaxis()

    # plot the reference
    ref_medval_arr = []
    ref_medval_arr_err = []
    for i in range(len(kron_frac)):

        sel = np.isfinite(master_ref_sb[f'aperture_sb_arcsec2_{i}']) & (master_sb[cvar_name] >= cvar_binmin) & (master_sb[cvar_name] <= cvar_binmax)
    
        ref_medval = np.median(master_ref_sb[f'aperture_sb_arcsec2_{i}'][sel] * master_ref_sb['photflam'][sel])
        ref_samples = bootstrap(master_ref_sb[f'aperture_sb_arcsec2_{i}'][sel] * master_ref_sb['photflam'][sel], bootfunc=np.median, bootnum=1000)
        ref_medval_mag = -2.5*np.log10(ref_medval) - 21.1
        ref_samples_mag = -2.5*np.log10(ref_samples) - 21.1
        ref_medval_mag_err = np.nanstd(ref_samples_mag)     
                
        ref_medval_arr.append(ref_medval_mag)
        ref_medval_arr_err.append(ref_medval_mag_err)
    
    ref_medval_arr = np.array(ref_medval_arr)
    ref_medval_arr_err = np.array(ref_medval_arr_err)  
    
    if rebin > 1:
        ref_medval_arr = block_reduce(ref_medval_arr, rebin, np.mean)
        ref_medval_arr_err = block_reduce(ref_medval_arr_err**2, rebin, np.sum)
        ref_medval_arr_err = np.sqrt(ref_medval_arr_err/rebin)
        kron_frac_plot = block_reduce(kron_frac, rebin, np.mean)
    else:
        kron_frac_plot = kron_frac
    
    if not dmag:
    
        ax.errorbar(kron_frac_plot, ref_medval_arr, ref_medval_arr_err, color='black',
                    label='truth')

    # plot the measurement images
    
    color_set = ['#e41a1c','#377eb8', '#4daf4a', 'magenta']
    binsize = kron_frac[1] - kron_frac[0]
    bin_xshift = np.linspace(-1,1,len(cvar_binsmin))*binsize*bin_shift_mag

    for lo, hi, color, shift in zip(cvar_binsmin, cvar_binsmax, color_set, bin_xshift):
        
        nsamples_arr = []
        medval_arr = []
        medval_arr_err = []
        
        in_bin = (master_sb[cvar_name] >= lo) & (master_sb[cvar_name] <= hi)
        
        # average these at each radius
        i_max = len(kron_frac)-1
        for i in range(len(kron_frac)):
            # select good data
            sel = in_bin & np.isfinite(master_sb[f'aperture_sb_arcsec2_{i}'])
            # take median brightness in counts/s/arcsec^2 and get bootstrapped vals
            medval = np.median(master_sb[f'aperture_sb_arcsec2_{i}'][sel] * master_sb['photflam'][sel])
            samples = bootstrap(master_sb[f'aperture_sb_arcsec2_{i}'][sel] * master_sb['photflam'][sel], bootfunc=np.median, bootnum=1000)
            # convert to mag/arcsec^2
            medval_mag = -2.5*np.log10(medval) - 21.1
            samples_mag = -2.5*np.log10(samples) - 21.1
            medval_mag_err = np.nanstd(samples_mag)
            
            # append to arrays
            nsamples_arr.append(np.sum(sel))
            medval_arr.append(medval_mag)
            medval_arr_err.append(medval_mag_err)
                   
        nsamples_arr = np.array(nsamples_arr)
        medval_arr = np.array(medval_arr)
        medval_arr_err = np.array(medval_arr_err)  
        
        print(len(medval_arr))
        
        if rebin > 1:
            medval_arr = block_reduce(medval_arr, rebin, np.mean)
            medval_arr_err = block_reduce(medval_arr_err**2, rebin, np.sum)
            medval_arr_err = np.sqrt(medval_arr_err/rebin)        
        # now plot
        
        if dmag:
            sb_dmag = medval_arr - ref_medval_arr 
            sb_dmag_err = np.sqrt(medval_arr_err**2 + ref_medval_arr_err**2)
            ax.errorbar(kron_frac_plot+shift, sb_dmag, sb_dmag_err, color=color,label='{} - {}'.format(lo, hi))
        else:
            ax.errorbar(kron_frac_plot, medval_arr, medval_arr_err, color=color,
                        label='{} - {}'.format(lo, hi))
        

    ax.legend(title='background [e-/s]')
    ax.set_xlabel('r / r$_{kron}$')
    if dmag:
        ax.set_ylabel('$\Delta$ mag arcsec$^{-2}$')
    else:
        ax.set_ylabel('mag arcsec$^{-2}$')
    if dmag:
        ax.axhline(0,lw=3,ls='--',alpha=0.7)

    
def plot_sbprofs(sbprof_files, line_labels, colors, lss, det_cat_file,
                 measurement_cat_files=None, max_mag=26.5, max_star_prob=0.9,
                 min_yshifts = 0, max_yshifts=2048, min_samples=10, dmag=True):
    
    if measurement_cat_files is None:
        measurement_cat_files = [None] * len(sbprof_files)
    
    kron_frac = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])  # find better way to do this so the kron fracs are not hard coded!
    
    fig, ax = plt.subplots()
    plt.gca().invert_yaxis()
    
    for file, line_label, color, ls, measurement_cat_file in zip(sbprof_files, line_labels, colors, lss, measurement_cat_files):
        sb = Table.read(file)
        
        # get corresponding image
        image_file = file.split('_sbprof')[0] + '.fits'
        photflam = fits.getval(image_file, 'PHOTFLAM')
        
        # load the det cat and measurement cat
        det_cat = Table.read(det_cat_file)
        
        # define good things
        det_snr = det_cat['kron_flux'] / det_cat['kron_fluxerr']
        star_prob = det_cat['star_prob']
        
        # define det image and get photflam
        det_image = det_cat_file.split('_photutils')[0] + '.fits'
        det_photflam = fits.getval(det_image, 'PHOTFLAM')
        det_mag = -2.5*np.log10(det_cat['kron_flux'] * det_photflam) - 21.1
        
        good_det = (det_snr > 10) & (det_mag < max_mag) & (star_prob < max_star_prob) & \
        (det_cat['xcentroid'] < (det_cat['ycentroid']/27.51+6417)) & \
        (det_cat['ycentroid'] > (-0.112*det_cat['xcentroid']+745.2)) & \
        (det_cat['ycentroid'] < (-0.089*det_cat['xcentroid'] + 6741.1)) & \
        (det_cat['xcentroid'] > (det_cat['ycentroid']/109.15+322.4))
        
        if measurement_cat_file is not None:
            
            meas_cat = Table.read(measurement_cat_file)
            
            # define good goods
            meas_snr = meas_cat['kron_flux']/meas_cat['kron_fluxerr']
            
            y_shifts = meas_cat['y_shifts']
            
            good_meas = (meas_snr > 5) & (y_shifts <= max_yshifts) & (y_shifts >= min_yshifts) & np.isfinite(y_shifts) & \
                        ((meas_cat['area']/meas_cat['segment_area']) > 0.9) & \
                        ((meas_cat['kron_area_unmasked'] / meas_cat['kron_area']) > 0.9) 
        
        if (measurement_cat_file is not None):
            good = good_det & good_meas
        else:
            good = good_det

            
        good_labels = det_cat['label'][good]
        
        # check for these labels in the sbprof table, reject any not in this list
        sb['keep'] = False
        for label in good_labels:
            sel = np.where(sb['label'] == label)[0]
            if len(sel) > 0:
                sb['keep'][sel] = True
        
        nkeep = np.sum(sb['keep'])
                
        print(f'keeping a total of {nkeep} out of {len(sb)} profiles with selection')
        sb = sb[sb['keep']]
    
        medval_arr = []
        medval_arr_err = []
        nsamples_arr = []
        
        i_max = len(kron_frac)-1
        for i in range(len(kron_frac)):
            # select good data
            sel = np.isfinite(sb[f'aperture_sb_arcsec2_{i}'])
            # take median brightness in counts/s/arcsec^2 and get bootstrapped vals
            medval = np.median(sb[f'aperture_sb_arcsec2_{i}'][sel])
            samples = bootstrap(sb[f'aperture_sb_arcsec2_{i}'][sel], bootfunc=np.median, bootnum=1000)
            # convert to mag/arcsec^2
            medval_mag = -2.5*np.log10(medval * photflam) - 21.1
            samples_mag = -2.5*np.log10(samples * photflam) - 21.1
            medval_mag_err = np.nanstd(samples_mag)
            
            # append to arrays
            nsamples_arr.append(np.sum(sel))
            medval_arr.append(medval_mag)
            medval_arr_err.append(medval_mag_err)
            
        nsamples_arr = np.array(nsamples_arr)
        medval_arr = np.array(medval_arr)
        medval_arr_err = np.array(medval_arr_err)    
        
        # plot
        sel = (nsamples_arr > min_samples) & (np.isfinite(medval_arr)) & (np.isfinite(medval_arr_err))
        ax.errorbar(kron_frac[sel], medval_arr[sel], medval_arr_err[sel], label=line_label, color=color, ls=ls)
        print(medval_arr)
        print(medval_arr_err)
    
    ax.legend()
    ax.set_xlabel('r / r$_{kron}$')
    ax.set_ylabel('mag arcsec$^{-2}$')
    
def standard_det_qual(det_cat, min_snr=10, max_star_prob=0.3,
                      min_mag=21.5, flux_var='kron_flux',
                      max_mag=26):
    '''standard quality cuts for source detection catalogs'''
    
    fluxerr_var = flux_var + 'err'

    # define good things
    det_snr = det_cat[flux_var] / det_cat[fluxerr_var]
    star_prob = det_cat['star_prob']
    det_mag = -2.5*np.log10(det_cat['kron_flux'] * det_cat.meta['PHOTFLAM']) - 21.1
    
    good_det = (det_snr > min_snr) & (star_prob < max_star_prob) & \
               (det_mag >= min_mag) & (det_mag <= max_mag) & \
               ((det_cat['kron_radius']*det_cat['semimajor_sigma']/4) > 2) & \
                (det_cat['xcentroid'] < (det_cat['ycentroid']/27.51+6417)) & \
                (det_cat['ycentroid'] > (-0.112*det_cat['xcentroid']+745.2)) & \
                (det_cat['ycentroid'] < (-0.089*det_cat['xcentroid'] + 6741.1)) & \
                (det_cat['xcentroid'] > (det_cat['ycentroid']/109.15+322.4)) & \
                ((det_cat['ycentroid'] > (43.8*det_cat['xcentroid'] - 141451.4)) | \
                 (det_cat['ycentroid'] < (44.5*det_cat['xcentroid'] - 155626.5)))

    return good_det

def standard_meas_qual(meas_cat, min_snr=3, max_yshifts=2048, 
                       min_yshifts=0, min_segment_area=0.99, 
                       min_kron_area=0.99, flux_var='kron_flux'):
    '''standard quality cuts for source measurement catalogs'''
    
    meas_snr = meas_cat[flux_var] / meas_cat[flux_var + 'err']
    y_shifts = meas_cat['y_shifts']

    good_meas = (meas_snr > min_snr) & \
            (np.isfinite(meas_cat['y_shifts'])) & \
            ((meas_cat['area']/meas_cat['segment_area']) > 0.99) & \
            ((meas_cat['kron_area_unmasked'] / meas_cat['kron_area']) > 0.99) 
    
    return good_meas
    
    
def merge_latexify(table1, table2, columns=None):
    '''convert an astropy table to a latex table'''

    lines = []
    for t1, t2 in zip(table1, table2):
        # set background columns
        minbg = int(t1['cvarmin'])
        maxbg = int(t1['cvarmax'])
        background = f'{minbg:d}-{maxbg:d}'
        mean1 = f'{t1["mean"]:.3f}' + '$\pm$' + f'{t1["mean_err"]:.3f}'
        r1= f'{t1["spearman_r"]:.2f}'
        p1= f'{t1["spearman_p"]:.2E}'
        p1=p1.replace('E', '$\\times 10^{') + '}$' # replace E with latex format
        p1=p1.replace('10^{-0', '10^{-')
        slope1 = f'{t1["theil_slope"]:.3f}' + '$\pm$' + f'{t1["theil_slope_err"]:.3f}'

        # same for table 2
        mean2 = f'{t2["mean"]:.3f}' + '$\pm$' + f'{t2["mean_err"]:.3f}'
        r2= f'{t2["spearman_r"]:.2f}'
        p2= f'{t2["spearman_p"]:.2E}'
        p2=p2.replace('E', '$\\times 10^{') + '}$' # replace E with latex format
        p2=p2.replace('10^{-0', '10^{-')
        slope2 = f'{t2["theil_slope"]:.3f}' + '$\pm$' + f'{t2["theil_slope_err"]:.3f}'        

        sep = ' & '
        line = background + sep + mean1 + sep + mean2 + sep + r1 + sep + r2 + sep + p1 + sep + p2 + sep + slope1 + sep + slope2 + '\\\\'
        print(line)   
    
def latexify(table, columns=None):
    '''convert an astropy table to a latex table'''

    lines = []
    for t in table:
        # set background columns
        minbg = int(t['cvarmin'])
        maxbg = int(t['cvarmax'])
        background = f'{minbg:d}-{maxbg:d}'
        mean = f'{t["mean"]:.3f}' + '$\pm$' + f'{t["mean_err"]:.3f}'
        r= f'{t["spearman_r"]:.2f}'
        p= f'{t["spearman_p"]:.2E}'
        p=p.replace('E', '$\\times 10^{') + '}$' # replace E with latex format
        p=p.replace('10^{-0', '10^{-')
        slope = f'{t["theil_slope"]:.3f}' + '$\pm$' + f'{t["theil_slope_err"]:.3f}'

        sep = ' & '
        line = background + sep + mean + sep + r + sep + p + sep + slope + '\\\\'
        print(line)

if __name__ == "__main__":
    
    print('Running tests:')
    test_file = '/Users/dstark/acs_work/cte/extended/data/processed/10325/10325_J91JC4010_drc_sci.fits'
    image = fits.getdata(test_file)
    header = fits.getheader(test_file)
    # print('Testing background map measurement')
    # bkg = calc_background_map(image, background_box_size=512, plot=True) 
    
    print('Testing Source Detection')
    cat, segment, bkg = measure_source_properties(image, header)
    
    print('Testing Source Detection when input catalog provided')
    test_file2 = '/Users/dstark/acs_work/cte/extended/data/processed/16870/16870_JERU01010_drc_sci_align.fits'
    image2 = fits.getdata(test_file2)
    header2 = fits.getheader(test_file2)
    cat2, segment2, bkg2 = measure_source_properties(image2, header2, 
                                                     deblend=True, 
                                                     input_segment_map=segment,
                                                     input_detection_cat=cat)
    
    
