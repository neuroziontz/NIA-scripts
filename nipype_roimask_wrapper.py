import os
import numpy as np
import nibabel as nib
from nipype.interfaces.base import TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec
from nipype.utils.filemanip import split_filename

class MaskMUSEROIInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='Path to label image', mandatory=True)
    roi_list = traits.List(traits.Int(), minlen=1, desc='Label indices to be combined', mandatory=True)

class MaskMUSEROIOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Mask')

class MaskMUSEROI(BaseInterface):
    """
    Generate mask of a composite ROI based on ROI index labels
    """

    input_spec = MaskMUSEROIInputSpec
    output_spec = MaskMUSEROIOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        _, base, _ = split_filename(in_file)

        # load MUSE labels file
        img = nib.load(in_file)
        img_data = img.get_fdata()
        mask_data = np.isin(img_data, self.inputs.roi_list)

        # Save results
        maskImg = nib.Nifti1Image(1*mask_data, img.affine)
        maskImgFile = base+'_maskedROI.nii.gz'
        nib.save(maskImg,maskImgFile)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        _, base, _ = split_filename(fname)

        outputs['out_file'] = os.path.abspath(base+'_maskedROI.nii.gz')

        return outputs
