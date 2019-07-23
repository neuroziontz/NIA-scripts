## Generates 3D ROI around maximum voxel in an image with specified morphology and size ##

import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from nipype.interfaces.base import TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec
from nipype.utils.filemanip import split_filename

class ROIcreateInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='Path to masked T-value image in functional space', mandatory=True)
    roi_name = traits.Str(desc='name of ROI being used to mask input image', mandatory=True)
    connectivity = traits.Int(desc='Shape of ROI corresponding to number of voxels squared from center that are included in ROI', mandatory=False) # look into traits.Enum
    size = traits.Int(desc='size of ROI corresponding to number of dilations performed on initial structure', mandatory=False)

class ROIcreateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Binary mask of seed ROI')

class ROIcreate(BaseInterface):
    """
    Create ROI of specified shape and size around a given voxel
    """

    input_spec = ROIcreateInputSpec
    output_spec = ROIcreateOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        _, base, _ = split_filename(in_file)

        # load input image into array
        out_filename = base+'_seed.nii.gz'
        img = nib.load(in_file)
        img_data = np.asanyarray(img.dataobj)

        # binarize mask if bilateral, otherwise draw seed ROI around peak voxel
        if 'bilateral' in self.inputs.roi_name:
            img_data_bin = np.where(img_data==0, 0, 1)
            img_bin = nib.Nifti1Image(img_data_bin, img.affine)
            nib.save(img_bin, out_filename)
        else:
            # get peak voxel in input image within mask
            peakvox_index = np.unravel_index(np.argmax(img_data, axis=None), img_data.shape)

            # create array of same shape with values of 0 for every voxel
            seed_data = np.zeros(img_data.shape[0:3])

            # assign a value of 1 to specified peak voxel
            seed_data[peakvox_index] = 1

            # generate ROI of specificed size and shape around peak voxel
            roi_struct = ndimage.generate_binary_structure(3, self.inputs.connectivity) # 3D ROI structure
            roi_data = ndimage.binary_dilation(seed_data, structure = roi_struct, iterations = self.inputs.size).astype(seed_data.dtype)

            # Save results
            roi_mask = nib.Nifti1Image(roi_data, img.affine)
            nib.save(roi_mask, out_filename)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)

        outputs['out_file'] = os.path.abspath(base+'_seed.nii.gz')

        return outputs
