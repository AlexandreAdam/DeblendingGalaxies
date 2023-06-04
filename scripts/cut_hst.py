from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

cutout_size = 128

filenames = ['hst_11602_02_wfc3_uvis_f814w_01_drz.fits',
            'hst_11602_02_wfc3_uvis_f814w_02_drz.fits',
            'hst_11602_02_wfc3_uvis_f814w_03_drz.fits',
            'hst_11602_02_wfc3_uvis_f814w_04_drz.fits',
            'hst_11602_02_wfc3_uvis_f814w_05_drz.fits',
            'hst_11602_02_wfc3_uvis_f814w_06_drz.fits']


img_set = []

tPlot, axes = plt.subplots(
        nrows=6, ncols=1, sharex=True, sharey=True,figsize=(6,4*6)
        )
for i, filename in enumerate(filenames):
    data_fits = fits.open(filename)
    print(data_fits.info())
    img = data_fits['PRIMARY'].data
    print(img.shape)

    axes[i].imshow(img,vmin=-0.1, vmax=0.1)
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])
    img_set.append(img)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('hst_snapshots.png')


cutouts = []
bad_cutouts = []
img_set_masked = []

cut_num = 0

for img in img_set:
    cutout_data = img
    img_masked = np.copy(img)
    #cutout_data = np.clip(cutout_data, -20., 20.)

    for i in range(img.shape[0]//cutout_size):
        for j in range(img.shape[1]//cutout_size):
            cutout = cutout_data[i*cutout_size:(i+1)*cutout_size, j*cutout_size:(j+1)*cutout_size]
            cut_num += 1

            if np.abs(np.sum(cutout))/cutout_size**2 < 0.04:
                if np.sum(cutout == 0.) == 0.:
                    if cutout.shape[1] == cutout_size and cutout.shape[0] == cutout_size:
                        cutouts.append(cutout)
            else:
                img_masked[i*cutout_size:(i+1)*cutout_size, j*cutout_size:(j+1)*cutout_size] = 0.
                if np.sum(cutout == 0.) == 0.:
                    #print('yo')
                    bad_cutouts.append(cutout)
    img_set_masked.append(img_masked)


img_set = []

tPlot, axes = plt.subplots(
        nrows=6, ncols=1, sharex=True, sharey=True,figsize=(6,4*6)
        )
for i, img in enumerate(img_set_masked):
    axes[i].imshow(img, vmin=-0.1, vmax=0.1)
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('hst_snapshots_masked.png')

#cutouts = np.array(cutouts)
#sums = np.sum(cutouts, axis=(1,2))/cutout_size**2



plt.figure(figsize=(8,6))
plt.imshow(bad_cutouts[563])
plt.colorbar()
plt.savefig('563.png')
#np.save('hst_inference_noise.npy', bad_cutouts[563])

plt.figure(figsize=(8,6))
plt.imshow(bad_cutouts[646], vmax=0.1, cmap='gray')
plt.colorbar()
plt.savefig('646.jpg', dpi=150)


frames = [] # for storing the generated images
fig = plt.figure()
ax = fig.add_subplot(111)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
for i in range(len(bad_cutouts)):
    mu = np.mean(bad_cutouts[i])
    std = np.std(bad_cutouts[i])
    ttl = plt.text(0.5, 1.01, '{}, mu={:.2f},sig={:.2f},min={:.2f},max={:.2f}'.format(i, mu, std, np.min(bad_cutouts[i]), np.max(bad_cutouts[i])), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

    
    img1 = plt.imshow(bad_cutouts[i],animated=True)
    #img1 = plt.hist(cutouts[i*10].flatten(),bins=30)
    frames.append([img1, ttl])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
ani.save('bad_cutouts.mp4')
#plt.show()

print("Number of cutouts: ", len(cutouts))
#np.save('hst_cutouts_noclip.npy', np.array(cutouts))


