"""
Deep learning plotting utilities

@author: Joseph Jennings
@version: 2021.04.19
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from .image import remove_colorbar


def plot_seglabel(
    img,
    lbl,
    show=True,
    color='red',
    fname=None,
    **kwargs,
) -> None:
  """
  Plots a binary label on top of an image

  Parameters:
    img   - image [nz,nx]
    lbl   - fault labels [nz,nx]
    show  - flag for showing the image [True]
    color - color of label to be plotted on image ['red']
    fname - name of output file to be saved (without the extension) [None]
  """
  [nz, nx] = img.shape
  if (img.shape != lbl.shape):
    raise Exception('Input image and label must be same size')
  # Get mask
  mask = np.ma.masked_where(lbl == 0, lbl)
  # Select colormap
  cmap = colors.ListedColormap([color, 'white'])
  fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
  ax = fig.add_subplot(111)
  # Plot image
  ox = kwargs.get('ox', 0.0)
  xmax = ox + kwargs.get('dx', 1.0) * nx
  oz = kwargs.get('oz', 0.0)
  zmax = oz + kwargs.get('dz', 1.0) * nz
  pclip = kwargs.get('pclip', 1.0)
  ax.imshow(
      img,
      cmap=kwargs.get('cmap', 'gray'),
      vmin=pclip * kwargs.get('vmin', np.min(img)),
      vmax=pclip * kwargs.get('vmax', np.max(img)),
      extent=[ox, xmax, zmax, oz],
      interpolation=kwargs.get("interp", "bilinear"),
  )
  ax.set_xlabel(kwargs.get('xlabel', 'X (km)'),
                fontsize=kwargs.get('labelsize', 14))
  ax.set_ylabel(kwargs.get('ylabel', 'Z (km)'),
                fontsize=kwargs.get('labelsize', 14))
  ax.set_title(kwargs.get('title', ''), fontsize=kwargs.get('labelsize', 14))
  ax.tick_params(labelsize=kwargs.get('ticksize', 14))
  if fname:
    ax.set_aspect(kwargs.get('aspect', 1.0))
    plt.savefig(fname + "-img.png",
                bbox_inches='tight',
                dpi=150,
                transparent=True)
  # Plot label
  ax.imshow(mask, cmap=cmap, extent=[ox, xmax, zmax, oz])
  ax.set_aspect(kwargs.get('aspect', 1.0))
  if show:
    plt.show()
  if fname:
    plt.savefig(fname + "-lbl.png",
                bbox_inches='tight',
                dpi=150,
                transparent=True)
    plt.close()


def plot_segprobs(
    img,
    prd,
    pmin=0.01,
    alpha=0.5,
    show=True,
    fname=None,
    **kwargs,
) -> None:
  """
  Plots unthresholded predictions on top of an image

  Parameters:
    img   - the input image [nz,nx]
    prd   - the predicted fault probability [nz,nx]
    pmin  - the minimum probability to display [0.01]
    alpha - transparency parameter for displaying probability [0.5]
    show  - flag for displaying the image [True]
  """
  [nz, nx] = img.shape
  if (img.shape != prd.shape):
    raise Exception('Input image and predictions must be same size')
  mask = np.ma.masked_where(prd <= pmin, prd)
  fig = plt.figure(figsize=(kwargs.get('wbox', 8), kwargs.get('hbox', 6)))
  ax = fig.add_subplot(111)
  # Plot image
  ox = kwargs.get('ox', 0.0)
  xmax = ox + kwargs.get('dx', 1.0) * nx
  oz = kwargs.get('oz', 0.0)
  zmax = oz + kwargs.get('dz', 1.0) * nz
  pclip = kwargs.get('pclip', 1.0)
  im = ax.imshow(
      img,
      cmap=kwargs.get('cmap', 'gray'),
      vmin=pclip * kwargs.get('vmin', np.min(img)),
      vmax=pclip * kwargs.get('vmax', np.max(img)),
      extent=[ox, xmax, zmax, oz],
      interpolation=kwargs.get("interp", "bilinear"),
  )
  ax.set_xlabel(kwargs.get('xlabel', 'X (km)'),
                fontsize=kwargs.get('labelsize', 18))
  if kwargs.get('zlabel', True):
    ax.set_ylabel(kwargs.get('ylabel', 'Z (km)'),
                  fontsize=kwargs.get('labelsize', 18))
  else:
    ax.axes.get_yaxis().set_visible(False)
  ax.set_title(kwargs.get('title', ''), fontsize=kwargs.get('labelsize', 18))
  ax.tick_params(labelsize=kwargs.get('ticksize', 18))
  # Set colorbar
  cbar_ax = fig.add_axes([
      kwargs.get('barx', 0.91),
      kwargs.get('barz', 0.12),
      kwargs.get('wbar', 0.02),
      kwargs.get('hbar', 0.75)
  ])
  cbar = fig.colorbar(im,
                      cbar_ax,
                      format='%.1f',
                      boundaries=np.arange(pmin, 1.1, 0.1))
  cbar.ax.tick_params(labelsize=kwargs.get('ticksize', 18))
  cbar.set_label(kwargs.get('barlabel', 'Fault probablility'),
                 fontsize=kwargs.get("barlabelsize", 18))
  if fname:
    ftype = kwargs.get('ftype', 'png')
    ax.set_aspect(kwargs.get('aspect', 1.0))
    plt.savefig(
        fname + "-img-tmp.png",
        bbox_inches='tight',
        dpi=150,
        transparent=True,
    )
  cbar.remove()
  # Plot label
  imp = ax.imshow(
      mask,
      cmap='jet',
      extent=[ox, xmax, zmax, oz],
      interpolation=kwargs.get("pinterp", "bilinear"),
      vmin=pmin,
      vmax=1.0,
      alpha=alpha,
  )
  ax.set_aspect(kwargs.get('aspect', 1.0))
  # Set colorbar
  cbar_axp = fig.add_axes([
      kwargs.get('barx', 0.91),
      kwargs.get('barz', 0.12),
      kwargs.get('wbar', 0.02),
      kwargs.get('hbar', 0.75)
  ])
  cbarp = fig.colorbar(imp, cbar_axp, format='%.1f')
  cbarp.ax.tick_params(labelsize=kwargs.get('ticksize', 18))
  cbarp.set_label(kwargs.get('barlabel', 'Fault probablility'),
                  fontsize=kwargs.get("barlabelsize", 18))
  cbarp.draw_all()
  if show:
    plt.show()
  if fname:
    plt.savefig(
        fname + "-prd." + ftype,
        bbox_inches='tight',
        dpi=150,
        transparent=True,
    )
    plt.close()
    # Crop and pad the image so they are the same size
    remove_colorbar(
        fname + "-img-tmp.png",
        cropsize=kwargs.get('cropsize', 0),
        oftype=ftype,
        opath=fname + "-img." + ftype,
    )
  if kwargs.get('return_handle', True):
    return fig, ax


def plot_segprobslabel(
    img,
    prd,
    lbl,
    pmin=0.01,
    alpha=0.5,
    color='red',
    show=True,
    fname=None,
    **kwargs,
) -> None:
  """
  Plots both the probability and the label on the image

  Parameters:
    img - the input image
    prd - the predicted probabilities
    lbl - the thresholded label
  """
  [nz, nx] = img.shape
  if img.shape != lbl.shape:
    raise Exception('Input image and predictions must be same size')
  # Get masks
  lbl_mask = np.ma.masked_where(lbl == 0, lbl)
  prd_mask = np.ma.masked_where(prd <= pmin, prd)
  # Select colormaps
  lbl_cmap = colors.ListedColormap([color, 'white'])
  fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
  ax = fig.add_subplot(111)
  # Plot image
  ox = kwargs.get('ox', 0.0)
  xmax = ox + kwargs.get('dx', 1.0) * nx
  oz = kwargs.get('oz', 0.0)
  zmax = oz + kwargs.get('dz', 1.0) * nz
  pclip = kwargs.get('pclip', 1.0)
  im = ax.imshow(
      img,
      cmap=kwargs.get('cmap', 'gray'),
      vmin=pclip * kwargs.get('vmin', np.min(img)),
      vmax=pclip * kwargs.get('vmax', np.max(img)),
      extent=[ox, xmax, zmax, oz],
      interpolation=kwargs.get("interp", "bilinear"),
  )
  ax.set_xlabel(kwargs.get('xlabel', 'X (km)'),
                fontsize=kwargs.get('labelsize', 18))
  ax.set_ylabel(kwargs.get('ylabel', 'Z (km)'),
                fontsize=kwargs.get('labelsize', 18))
  ax.set_title(kwargs.get('title', ''), fontsize=kwargs.get('labelsize', 18))
  ax.tick_params(labelsize=kwargs.get('ticksize', 18))
  # Set colorbar
  cbar_ax = fig.add_axes([
      kwargs.get('barx', 0.91),
      kwargs.get('barz', 0.12),
      kwargs.get('wbar', 0.02),
      kwargs.get('hbar', 0.75)
  ])
  cbar = fig.colorbar(
      im,
      cbar_ax,
      format='%.1f',
      boundaries=np.arange(pmin, 1.1, 0.1),
  )
  cbar.ax.tick_params(labelsize=kwargs.get('ticksize', 18))
  cbar.set_label(kwargs.get('barlabel', 'Fault probablility'),
                 fontsize=kwargs.get("barlabelsize", 18))
  if fname:
    ftype = kwargs.get('ftype', 'png')
    ax.set_aspect(kwargs.get('aspect', 1.0))
    plt.savefig(
        fname + "-img-tmp.png",
        bbox_inches='tight',
        dpi=150,
        transparent=True,
    )
  cbar.remove()
  # Plot label
  ax.imshow(lbl_mask, cmap=lbl_cmap, extent=[ox, xmax, zmax, oz])
  # Plot predictions
  imp = ax.imshow(
      prd_mask,
      cmap='jet',
      extent=[ox, xmax, zmax, oz],
      interpolation=kwargs.get("pinterp", "bilinear"),
      vmin=pmin,
      vmax=1.0,
      alpha=alpha,
  )
  ax.set_aspect(kwargs.get('aspect', 1.0))
  # Set colorbar
  cbar_axp = fig.add_axes([
      kwargs.get('barx', 0.91),
      kwargs.get('barz', 0.12),
      kwargs.get('wbar', 0.02),
      kwargs.get('hbar', 0.75)
  ])
  cbarp = fig.colorbar(imp, cbar_axp, format='%.1f')
  cbarp.ax.tick_params(labelsize=kwargs.get('ticksize', 18))
  cbarp.set_label(kwargs.get('barlabel', 'Fault probablility'),
                  fontsize=kwargs.get("barlabelsize", 18))
  cbarp.draw_all()
  if show:
    plt.show()
  if fname:
    plt.savefig(
        fname + "-prd." + ftype,
        bbox_inches='tight',
        dpi=150,
        transparent=True,
    )
    plt.close()
    # Crop and pad the image so they are the same size
    remove_colorbar(
        fname + "-img-tmp.png",
        cropsize=kwargs.get('cropsize', 0),
        oftype=ftype,
        opath=fname + "-img." + ftype,
    )


def plot_patchgrid2d(
    img,
    nzp,
    nxp,
    strdz=None,
    strdx=None,
    dz=None,
    dx=None,
    oz=None,
    ox=None,
    transp=False,
    pltcoords=True,
    **kwargs,
) -> None:
  """
  Plots the patch grid on the input image

  Parameters:
    img       - the input image [nz,nx]
    nzp       - the size of the patch in z (samples)
    nxp       - the size of the patch in x (samples)
    strdz     - patch stride in z (samples) [nzp//2]
    strdx     - patch stride in x (samples) [nxp//2]
    dz        - depth sampling of image [1.0]
    dx        - lateral sampling of image [1.0]
    oz        - depth origin of image [0.0]
    ox        - lateral origin of image [0.0]
    transp    - flag indicating to transpose the input image [False]
    pltcoords - flag indicating to plot the patch
                coordinates on the image [True]
  """
  # Get the image axes
  if (transp):
    nx, nz = img.shape
  else:
    nz, nx = img.shape
  if (dz is None):
    dz = 1.0
  if (oz is None):
    oz = 0.0
  if (dx is None):
    dx = 1.0
  if (ox is None):
    ox = 0.0

  if (strdz is None):
    strdz = nzp // 2
  if (strdx is None):
    strdx = nxp // 2

  # Make the patch grids
  bgz = 0
  egz = nz * dz + 1
  dgz = nzp * dz
  bgx = 0
  egx = nx * dx + 1
  dgx = nxp * dx

  # Get number of patches in each dimensions
  nptchz, remz = divmod(nz, nzp)
  nptchx, remx = divmod(nx, nxp)

  # Plotting parameters
  cmap = kwargs.get('cmap', 'gray')
  pclip = kwargs.get('pclip', 1.0)
  vmin = kwargs.get('vmin', pclip * np.min(img))
  vmax = kwargs.get('vmax', pclip * np.max(img))
  xlabel = kwargs.get('xlabel', 'X (km)')
  zlabel = kwargs.get('zlabel', 'Z (km)')
  fsize = kwargs.get('fsize', 15)
  interp = kwargs.get('interp', 'bilinear')
  aspect = kwargs.get('aspect', 'auto')
  xmin = ox
  xmax = ox + nx * dx
  zmin = oz
  zmax = oz + nz * dz
  textsize = kwargs.get('textsize', 12)
  if (strdx != 0):
    xshft = 2
  if (strdz != 0):
    zshft = 2

  fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
  ax = fig.gca()
  zticks = np.arange(bgz, egz, dgz)
  xticks = np.arange(bgx, egx, dgx)
  ax.set_xticks(xticks)
  ax.set_yticks(zticks)
  ax.imshow(img,
            cmap=cmap,
            interpolation=interp,
            vmin=vmin,
            vmax=vmax,
            extent=[xmin, xmax, zmax, zmin],
            aspect=aspect)
  ax.grid(linestyle='-', color='k', linewidth=2)
  ax.set_ylabel(zlabel, fontsize=fsize)
  ax.set_xlabel(xlabel, fontsize=fsize)
  ax.tick_params(labelsize=fsize)
  tot1 = nptchz * nptchx
  ax.set_title(
      r'Grid 1: X-stride=0, Z-stride=0 $\rightarrow \,\, %d\times%d = %d$ patches'
      % (nptchx, nptchz, tot1),
      fontsize=fsize)
  # Plot the coordinates
  if (pltcoords):
    plot_patchcoords(nptchz, nptchx, zticks, xticks, zmax, xmax, 0, 0, zshft,
                     xshft, textsize, 'k')

  if (strdx != 0):
    fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
    ax = fig.gca()
    zticks = np.arange(bgz, egz, dgz)
    xticks = np.arange(bgx + strdx * dx, egx - strdx * dx, dgx)
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.imshow(img,
              cmap=cmap,
              interpolation=interp,
              vmin=vmin,
              vmax=vmax,
              extent=[xmin, xmax, zmax, zmin],
              aspect=aspect)
    ax.grid(which='major', linestyle='-', color='r', linewidth=2)
    ax.set_ylabel(zlabel, fontsize=fsize)
    ax.set_xlabel(xlabel, fontsize=fsize)
    ax.tick_params(labelsize=fsize)
    tot2 = (nptchx - 1) * nptchz
    ax.set_title(
        r'Grid 2: X-stride=%d, Z-stride=0 $\rightarrow \,\, %d\times%d = %d$ patches'
        % (strdx, nptchx - 1, nptchz, tot2),
        fontsize=fsize)
    if (pltcoords):
      plot_patchcoords(nptchz, nptchx - 1, zticks, xticks, zmax, xmax, 0, 1,
                       zshft, xshft, textsize, 'r')

  if (strdz != 0):
    fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
    ax = fig.gca()
    zticks = np.arange(bgz + strdz * dz, egz - strdz * dz, dgz)
    xticks = np.arange(bgx, egx, dgx)
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.imshow(img,
              cmap=cmap,
              interpolation=interp,
              vmin=vmin,
              vmax=vmax,
              extent=[xmin, xmax, zmax, zmin],
              aspect=aspect)
    ax.grid(which='major', linestyle='-', color='g', linewidth=2)
    ax.set_ylabel(zlabel, fontsize=fsize)
    ax.set_xlabel(xlabel, fontsize=fsize)
    ax.tick_params(labelsize=fsize)
    tot3 = (nptchx) * (nptchz - 1)
    ax.set_title(
        r'Grid 3: X-stride=0, Z-stride=%d $\rightarrow \,\, %d\times%d = %d$ patches'
        % (strdz, nptchx, nptchz - 1, tot3),
        fontsize=fsize)
    if (pltcoords):
      plot_patchcoords(nptchz - 1, nptchx, zticks, xticks, zmax, xmax, 1, 0,
                       zshft, xshft, textsize, 'g')

  if (strdx != 0 and strdz != 0):
    fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 6)))
    ax = fig.gca()
    zticks = np.arange(bgz + strdz * dz, egz - strdz * dz, dgz)
    xticks = np.arange(bgx + strdx * dx, egx - strdx * dx, dgx)
    ax.set_xticks(xticks)
    ax.set_yticks(zticks)
    ax.imshow(img,
              cmap=cmap,
              interpolation=interp,
              vmin=vmin,
              vmax=vmax,
              extent=[xmin, xmax, zmax, zmin],
              aspect=aspect)
    ax.grid(which='major', linestyle='-', color='b', linewidth=2)
    ax.set_ylabel(zlabel, fontsize=fsize)
    ax.set_xlabel(xlabel, fontsize=fsize)
    ax.tick_params(labelsize=fsize)
    tot4 = (nptchx - 1) * (nptchz - 1)
    ax.set_title(
        r'Grid 4: X-stride=%d, Z-stride=%d $\rightarrow \,\, %d\times%d = %d$ patches'
        % (strdx, strdz, nptchx - 1, nptchz - 1, tot4),
        fontsize=fsize)
    if (pltcoords):
      plot_patchcoords(
          nptchz - 1,
          nptchx - 1,
          zticks,
          xticks,
          zmax,
          xmax,
          1,
          1,
          zshft,
          xshft,
          textsize,
          'b',
      )

  if kwargs.get('totptchsqc', None) is not None:
    totptchs = kwargs.get('totptchsqc').shape[0]
  totptchs = tot1 + tot2 + tot3 + tot3
  print("Total number of patches: %d = %d + %d + %d + %d" %
        (totptchs, tot1, tot2, tot3, tot4))

  plt.show()


def plot_patchcoords(nptchz, nptchx, zticks, xticks, zmax, xmax, zidxi, xidxi,
                     zshft, xshft, textsize, color):
  """
  Plots the patch coordinates on the image
  To be used with plot_patchgrid2d
  """
  idx = zticks < zmax
  zticksw = zticks[idx]
  idx = xticks < xmax
  xticksw = xticks[idx]
  # Get the offset
  xtickdx = xticksw[1] - xticksw[0]
  ztickdx = zticksw[1] - zticksw[0]
  for iztick in range(nptchz):
    zpos = zticksw[iztick] + ztickdx / 2
    xidx = xidxi
    for ixtick in range(nptchx):
      xpos = xticksw[ixtick] + xtickdx / 2
      plt.text(xpos,
               zpos,
               '(%d,%d)' % (zidxi, xidx),
               fontsize=textsize,
               color=color)
      xidx += xshft
    zidxi += zshft
