import time
import astropy.wcs as pywcs
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
myfontsize=22
plt.rcParams.update({'font.size': myfontsize})
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,ScalarFormatter, NullFormatter,MaxNLocator, NullLocator,LogLocator
from grizli.utils import get_line_wavelengths 
from grizli import utils



def make_random_cmap(ncolors=256, seed=None):
    """
    Make a matplotlib colormap consisting of (random) muted colors.

    A random colormap is very useful for plotting segmentation images.

    Parameters
    ----------
    ncolors : int, optional
        The number of colors in the colormap.  The default is 256.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.
        Separate function calls with the same ``seed`` will generate the
        same colormap.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
        The matplotlib colormap with random colors in RGBA format.
    """
    from matplotlib import colors

    rng = np.random.default_rng(seed)
    hue = rng.uniform(low=0.0, high=1.0, size=ncolors)
    sat = rng.uniform(low=0.2, high=0.7, size=ncolors)
    val = rng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((hue, sat, val))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    return colors.ListedColormap(colors.to_rgba_array(rgb))

def get_cmap(segmap, background_color='#000000ff', seed=None):
        
        """
        Define a matplotlib colormap consisting of (random) muted
        colors.

        This is useful for plotting the segmentation array.

        Parameters
        ----------
        

        background_color : Matplotlib color, optional
            The color of the first color in the colormap.
            The color may be specified using any of
            the `Matplotlib color formats
            <https://matplotlib.org/stable/tutorials/colors/colors.html>`_.
            This color will be used as the background color (label = 0)
            when plotting the segmentation image. The default color is
            black with alpha=1.0 ('#000000ff').

        seed : int, optional
            A seed to initialize the `numpy.random.BitGenerator`. If
            `None`, then fresh, unpredictable entropy will be pulled
            from the OS. Separate function calls with the same ``seed``
            will generate the same colormap.

        Returns
        -------
        cmap : `matplotlib.colors.ListedColormap`
            The matplotlib colormap with colors in RGBA format.
        """
        
        
        from matplotlib import colors
       
        #ncolors : int
        #    The number of the colors in the colormap.
        ncolors=np.max(segmap)+1 #len(np.unique(segmap))+1
        cmap = make_random_cmap(ncolors, seed=seed)

        #if background_color is not None:
        cmap.colors[0] = colors.to_rgba(background_color)
        #print(cmap.colors)
        return cmap
    
    
"""
Function for reading emission/absorption lines, their line fluxes (line flux errors), and equivalent widths
given the threshold in signal-to-noise ratio.
"""
def get_lines(full=None,min_snr=5):
    """
    full: output fits file name from grizli (*full.fits)
    min_snr: minimum signal-to-noise ratio of emission lines to be read from file
    """

       
    
    matching_keys = [key for key in full['COVAR'].header if key.startswith('FLUX_')]
    nlines = len(matching_keys)# full['PRIMARY'].header['NUMLINES']
    line_wave_obs_dict={}
    lines_snr_dict={}
    lines_flux_dict={}
    lines_fluxerr_dict={}
    EW16_dict={}
    EW50_dict={}
    EW84_dict={}
    EW_snr_dict={}
    lines_name=[]
    #lin
    _line_wavelengths, _line_ratios = get_line_wavelengths()
    for i in range(nlines):
        lineflux=full['COVAR'].header['FLUX_%s' % str(i).zfill(3)]
        lineflux_err=full['COVAR'].header['ERR_%s' % str(i).zfill(3)]
        comments=(full['COVAR'].header.comments['FLUX_%s'  % str(i).zfill(3) ]).split()
        if lineflux_err>0:
            _snr_line=lineflux/lineflux_err
        if lineflux_err==0:
            _snr_line=-99
    
        ew16=full['COVAR'].header['EW16_%s' % str(i).zfill(3)]
        ew50=full['COVAR'].header['EW50_%s' % str(i).zfill(3)]
        ew84=full['COVAR'].header['EW84_%s' % str(i).zfill(3)]
        if (_snr_line>min_snr) & (ew50>0):
            line_wave_obs_dict['%s' % comments[0]]=_line_wavelengths['%s' % comments[0]][0]*(1+full['ZFIT_STACK'].header['Z_MAP'])
            lines_snr_dict['%s' % comments[0]]=lineflux/lineflux_err
            lines_flux_dict['%s' % comments[0]]=lineflux
            lines_fluxerr_dict['%s' % comments[0]]=lineflux_err
            # compute 16th, 50th, and 84th percentiles of the distribution of equivalent width  
            EW16_dict['%s' % comments[0]]=full['COVAR'].header['EW16_%s' % str(i).zfill(3)]
            EW50_dict['%s' % comments[0]]=full['COVAR'].header['EW50_%s' % str(i).zfill(3)]
            EW84_dict['%s' % comments[0]]=full['COVAR'].header['EW84_%s' % str(i).zfill(3)]
            EW_snr_dict['%s' % comments[0]]=ew50/((ew84-ew16)/2)
            lines_name.append(comments[0])
    
    lines_prop_dicts={'name':lines_name,
                      'wavelength_obs':line_wave_obs_dict,
                      'flux':lines_flux_dict,
                      'flux_err':lines_fluxerr_dict,
                      'snr_line':lines_snr_dict,
                      'snr_ew':EW_snr_dict,
                      'ew_16':EW16_dict,
                      'ew_50':EW50_dict,
                      'ew_84':EW84_dict}
    return lines_prop_dicts
def show_drizzled_lines(line_hdu, full_line_list=['OII', 'Hb', 'OIII', 'Ha+NII', 'Ha', 'SII', 'SIII'], 
                        size_arcsec=2, cmap='plasma_r',scale=1., dscale=1, 
                        direct_filter=['F140W', 'F160W', 'F125W', 'F105W', 'F110W', 'F098M']):
    
    """Make a figure with the drizzled line maps
    
    Parameters
    ----------
    line_hdu : `~astropy.io.fits.HDUList`
        Result from `~grizli.multifit.MultiBeam.drizzle_fit_lines`
    
    full_line_list : list
        Line species too always show
    
    size_arcsec : float
        Thumbnail size in arcsec
    
    cmap : str
        colormap string
    
    scale : float
        Scale factor for line panels
    
    dscale : float
        Scale factor for direct image panel
        
    direct_filter : list
        Filter preference to show in the direct image panel.  Step through
        and stop if the indicated filter is available.
    
    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Figure object
        
    """
    
    show_lines = []
    print('full_line_list',full_line_list)
    for line in full_line_list:
        if line in line_hdu[0].header['HASLINES'].split():
            show_lines.append(line)

    if full_line_list == 'all':
        show_lines = line_hdu[0].header['HASLINES'].split()

    #print(line_hdu[0].header['HASLINES'], show_lines)

    # Dimensions
    line_wcs = pywcs.WCS(line_hdu['DSCI'].header)
    pix_size = utils.get_wcs_pscale(line_wcs)
    #pix_size = np.abs(line_hdu['DSCI'].header['CD1_1']*3600)
    majorLocator = MultipleLocator(1.)  # /pix_size)
    N = line_hdu['DSCI'].data.shape[0]/2

    crp = line_hdu['DSCI'].header['CRPIX1'], line_hdu['DSCI'].header['CRPIX2']
    crv = line_hdu['DSCI'].header['CRVAL1'], line_hdu['DSCI'].header['CRVAL2']
    imsize_arcsec = line_hdu['DSCI'].data.shape[0]*pix_size
    # Assume square
    sh = line_hdu['DSCI'].data.shape
    dp = -0.5*pix_size  # FITS reference is center of a pixel, array is edge
    dp = 0
    extent = (-imsize_arcsec/2.-dp, imsize_arcsec/2.-dp,
              -imsize_arcsec/2.-dp, imsize_arcsec/2.-dp)

    NL = len(show_lines)

    xsize = 3*(NL+1)
    fig = plt.figure(figsize=[xsize, 3.6])

    # Direct
    ax = fig.add_subplot(1, NL+1, 1)

    dext = 'DSCI'
    # Try preference for direct filter
    for filt in direct_filter:
        if ('DSCI', filt) in line_hdu:
            dext = 'DSCI', filt
            break

    ax.imshow(line_hdu[dext].data*dscale, vmin=-0.02, vmax=0.6, cmap=cmap, origin='lower', extent=extent)

    ax.set_title('Direct   {0}    z={1:.3f}'.format(line_hdu[0].header['ID'], line_hdu[0].header['REDSHIFT']))

    if 'FILTER' in line_hdu[dext].header:
        ax.text(0.03, 0.97, line_hdu[dext].header['FILTER'],
                transform=ax.transAxes, ha='left', va='top', fontsize=8)

    #ax.set_xlabel('RA')
    #ax.set_ylabel('Decl.')
    
    # Compass
    cosd = np.cos(line_hdu['DSCI'].header['CRVAL2']/180*np.pi)
    dra = np.array([1.5, 1,0,0,0])/3600.*0.12*size_arcsec/cosd
    dde = np.array([0, 0,0,1,1.5])/3600.*0.12*size_arcsec
    cx, cy = line_wcs.all_world2pix(crv[0]+dra, crv[1]+dde, 0)
    cx = (cx-cx.max())*pix_size
    cy = (cy-cy.max())*pix_size
    c0 = 0.95*size_arcsec
    ax.plot(cx[1:-1]+c0, cy[1:-1]+c0,
            linewidth=1, color='0.5')
    ax.text(cx[0]+c0, cy[0]+c0, r'$E$',
            ha='center', va='center', fontsize=7, color='0.5')
    ax.text(cx[4]+c0, cy[4]+c0, r'$N$',
            ha='center', va='center', fontsize=7, color='0.5')
    
    # 1" ticks
    ax.errorbar(-0.5, -0.9*size_arcsec, yerr=0, xerr=0.5, color='k')
    ax.text(-0.5, -0.9*size_arcsec, r'$1^{\prime\prime}$', ha='center', va='bottom', color='k')

    # Line maps
    for i, line in enumerate(show_lines):
        ax = fig.add_subplot(1, NL+1, 2+i)
        ax.imshow(line_hdu['LINE', line].data*scale, vmin=-0.02,
                  vmax=0.6, cmap=cmap, origin='lower', extent=extent)
        ax.set_title(r'%s %.3f $\mu$m' % (line, line_hdu['LINE', line].header['WAVELEN']/1.e4))

    # End things
    for ax in fig.axes:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlim(np.array([-1, 1])*size_arcsec)
        ax.set_ylim(np.array([-1, 1])*size_arcsec)

        #x0 = np.mean(ax.get_xlim())
        #y0 = np.mean(ax.get_xlim())
        ax.scatter(0, 0, marker='+', color='k', zorder=100, alpha=0.5)
        ax.scatter(0, 0, marker='+', color='w', zorder=101, alpha=0.5)

        ax.xaxis.set_major_locator(majorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    fig.tight_layout(pad=0.1, w_pad=0.5)
    fig.text(1-0.015*12./xsize, 0.02, time.ctime(), ha='right', va='bottom',
             transform=fig.transFigure, fontsize=5)

    return fig

def get_2dspectra(hdu=None,axs_f115w=None,
              axs_f150w=None,axs_f200w=None,cmap='viridis_r'):
    fs_offset = 0# subtitle fontsize offset
    h0 = hdu[0].header
    NX = h0['NGRISM']
    NY = 0
  
    grisms = OrderedDict()
    axs_allbands=[axs_f115w,axs_f150w,axs_f200w]
    for ig in range(NX):
        g = h0['GRISM{0:03d}'.format(ig+1)]
        NY = np.maximum(NY, h0['N'+g])
        grisms[g] = h0['N'+g]

    for g in grisms: # loop over filter
        if g =='F115W':
            ig=0
        if g =='F150W':
            ig=1
        if g=='F200W':
            ig=2
        iters=range(grisms[g])

        sci_i = hdu['SCI', g]
        wht_i = hdu['WHT', g]

        model_i = hdu['MODEL', g]

        #kern_i = hdu['KERNEL', g]
        h_i = sci_i.header
        clip = wht_i.data > 0
        if clip.sum() == 0:
            clip = np.isfinite(wht_i.data)

        avg_rms = 1/np.median(np.sqrt(wht_i.data[clip]))
        vmax = np.maximum(1.1*np.nanpercentile(sci_i.data[clip], 98),
                         5*avg_rms)
        # Spectrum
        sh = sci_i.data.shape
        extent = [h_i['WMIN'], h_i['WMAX'], 0, sh[0]]
        #sci 2D spectra (all PAs) (background subtracted by default)
        axs_allbands[ig][0,2].imshow(sci_i.data,origin='lower', interpolation='Nearest',
                  vmin=-0.1*vmax, vmax=vmax, cmap=cmap,
                  extent=extent, aspect='auto')

        axs_allbands[ig][0,2].set_title('%s' % (sci_i.header['EXTVER']),fontsize=myfontsize-fs_offset)
        #model continuum 2D spectra (all PAs)
        axs_allbands[ig][2,2].imshow(model_i.data,origin='lower', interpolation='Nearest',
                  vmin=-0.1*vmax, vmax=vmax, cmap=cmap,
                  extent=extent, aspect='auto')       
        # continuum subtracted and contamination subtracted
        #axs_allbands[ig][3,2].imshow(sci_i.data-model_i.data,origin='lower', interpolation='Nearest',
        #          vmin=-0.1*vmax, vmax=vmax, cmap=cmap,
        #          extent=extent, aspect='auto')
        axs_allbands[ig][3,2].axis('off')
        for k in [0,1,2,3]:
                axs_allbands[ig][k,2].set_yticklabels([])
                #axs_allbands[ig][k,2].set_xticklabels([])
                axs_allbands[ig][k,2].xaxis.set_major_locator(MultipleLocator(0.1))

        axs_allbands[ig][1,2].axis('off')
        all_PAs=[]
        for col,ip in enumerate(iters): # loop over pa for each filter
            #print(ip, ig)
            pa = h0['{0}{1:02d}'.format(g, ip+1)]
            all_PAs.append(pa)
            #print(ip, g,pa)
            sci_i = hdu['SCI', '{0},{1}'.format(g, pa)]
            wht_i = hdu['WHT', '{0},{1}'.format(g, pa)]
            contam_i = hdu['CONTAM', '{0},{1}'.format(g, pa)]
            model_i = hdu['MODEL', '{0},{1}'.format(g, pa)]
            h_i = sci_i.header
            sh = sci_i.data.shape
            extent = [h_i['WMIN'], h_i['WMAX'], 0, sh[0]]
            #print('pa',pa)

            # sci 2d spectra (not contamination subtract )
            axs_allbands[ig][0,col].imshow(sci_i.data,origin='lower', interpolation='Nearest',
                  vmin=-0.1*vmax, vmax=vmax, cmap=cmap,
                  extent=extent, aspect='auto')


            axs_allbands[ig][0,col].set_title('%s' % (sci_i.header['EXTVER']),fontsize=myfontsize-fs_offset)
            # model contamination 
            axs_allbands[ig][1,col].imshow(contam_i.data,origin='lower', interpolation='Nearest',
                  vmin=-0.1*vmax, vmax=vmax, cmap=cmap,
                  extent=extent, aspect='auto')

            #model continuum
            axs_allbands[ig][2,col].imshow(model_i.data,origin='lower', interpolation='Nearest',
                  vmin=-0.1*vmax, vmax=vmax, cmap=cmap,
                  extent=extent, aspect='auto')

            # sci - contamination - continuum model (left with emission line, if any)
            axs_allbands[ig][3,col].imshow(sci_i.data-contam_i.data-model_i.data,origin='lower', interpolation='Nearest',
                  vmin=-0.1*vmax, vmax=vmax, cmap=cmap,
                  extent=extent, aspect='auto')            
        axs_allbands[ig][0,0].text(-0.2,0.4,'science',transform=axs_allbands[ig][0,0].transAxes,fontsize=myfontsize-fs_offset,rotation=90) #  
        axs_allbands[ig][1,0].text(-0.4,0.0,'modeled\n contamination',transform=axs_allbands[ig][1,0].transAxes,fontsize=myfontsize-fs_offset,rotation=90) #  
        axs_allbands[ig][2,0].text(-0.4,0.0,'modeled\n continuum',transform=axs_allbands[ig][2,0].transAxes,fontsize=myfontsize-fs_offset,rotation=90) #
        axs_allbands[ig][3,0].text(-0.4,0.,'continuum\n subtracted',transform=axs_allbands[ig][3,0].transAxes,fontsize=myfontsize-fs_offset,rotation=90) #

        for kk,pa in enumerate(all_PAs):
            #print('kk,pa',kk,pa)
            #try:
            #_=np.shape(hdu['SCI','%s,%s' % (g,pa)])
            for jj in [0,1,2,3]:
                axs_allbands[ig][jj,kk].set_yticklabels([])
                axs_allbands[ig][jj,kk].xaxis.set_major_locator(MultipleLocator(0.1))
        if len(all_PAs)<2:
            for jj in [0,1,2,3]:
                axs_allbands[ig][jj,1].axis('off')

def scale_bar(ax, d, dist=1/0.13, text='1"', color='black', flipped=False, fontsize=15,linewidth=2):
    if flipped:
        p0 = d - d / 15. - dist
        p1 = d / 15.
        ax.plot([p0, p0 + dist], [p1, p1], linewidth=linewidth, color=color)
        ax.text(p0 + dist / 2., p1 + 0.02 * d, text, fontsize=fontsize, color=color, ha='center')
    else:
        p0 = d / 15.
        ax.plot([p0, p0 + dist], [p0, p0], linewidth=linewidth, color=color)
        ax.text(p0 + dist / 2., p0 + 0.02 * d, text, fontsize=fontsize, color=color, ha='center')
