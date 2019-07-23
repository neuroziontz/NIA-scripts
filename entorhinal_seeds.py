### Analysis of association of entorhinal cortex functional connectivity with entorhinal AV1451 signal ###

import math, sys, os, logging
import pandas as pd
import numpy as np
import nibabel as nib
from scipy import stats
from glob import glob
from nipype_roimask_wrapper import MaskMUSEROI
from roicreate_wrapper import ROIcreate

# nipype
from nipype.algorithms import misc
import nipype.interfaces.io as nio
from nipype.interfaces import fsl, afni, freesurfer, spm, ants
from nipype.pipeline.engine import Workflow, Node, JoinNode, MapNode
from nipype.interfaces.utility import Function, IdentityInterface, Merge
from nipype import config, logging
config.enable_debug_mode()
config.update_config({'execution': {'stop_on_first_crash': True}})
logging.update_logging(config)

# number of parallel processes
n_procs = 8

# directory to store the workflow results
output_dir = '/cog/home/jacob/entorhinal-connectivity-and-tau/nipype_output_longitudinal'
# directory with subdirectories containing processed AV1451 scans
images_dir = '/cog/home/jacob/entorhinal-connectivity-and-tau/data'
# directory with BLSA MUSE MNI152 label image
muse_img = '/cog/home/jacob/entorhinal-connectivity-and-tau/atlas_blsa_muse_mni152.nii.gz'
# mask files
maskfile = '/cog/home/jacob/entorhinal-connectivity-and-tau/BLSA_SPGR+MPRAGE_AllBaselines_Age60-80_Random100_averagetemplate_short_rMNI152_reoriented_brainmask.nii'
rs_template = '/cog/home/jacob/entorhinal-connectivity-and-tau/EPI.nii'

# Use standalone SPM rather than through MATLAB
standalone_spm_dir = '/cog/software/standalone_spm'
# Set up standalone SPM
matlab_cmd = os.path.join(standalone_spm_dir,'spm12','run_spm12.sh') + ' ' +  \
             os.path.join(standalone_spm_dir,'MCR','v713' + ' script')
spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)
spm.SPMCommand().version
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# spreadsheet with blsaid, blsavi, and image path for participants in sample
sample_spreadsheet = '/cog/home/jacob/entorhinal-connectivity-and-tau/data/taurs_long_sample.xlsx'
# values to be treated as missing in the spreadsheet - do not include NA as a null value as it is a valid EMSID
NAN_VALUES = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A','N/A', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan','']

# Read in the organization spreadsheet
data_table = pd.read_excel(sample_spreadsheet, keep_default_na=True, na_values=NAN_VALUES)

subjID = data_table['blsaid'].values.tolist()
visNo = data_table['blsavi'].values.tolist()
blsavi_integer = [ math.floor(x) for x in visNo ]
idvi_list = ["%04d_%02d-0" % idvi for idvi in zip(subjID,blsavi_integer)]

av1451_list = [None]*len(idvi_list)
for jj, idvi in enumerate(idvi_list):
    av1451file = os.path.join(images_dir,'av1451_images','rBLSA_'+idvi+'_P2_av1451_c3_mean_rbv_pvc_suvr_mni.nii.gz')
    if os.path.isfile(av1451file):
        av1451_list[jj] = av1451file
    else:
        raise FileNotFoundError(av1451file + " does not exist") # changed rBLSA_5600_16 file name --> rBLSA_5600_15 because mislabeled in PETstatus spreadsheet

rs_visNo = data_table['rs_blsavi'].values.tolist()
rs_blsavi_integer = [ math.floor(x) for x in rs_visNo ]
rs_idvi_list = ["%04d_%02d-0" % idvi for idvi in zip(subjID,rs_blsavi_integer)]

rsfmri_list = [None]*len(rs_idvi_list)
for jj, idvi in enumerate(rs_idvi_list):
    rsfile = os.path.join(images_dir,'rsfmri_images','BLSA_'+idvi+'_10','remeaned_filreg_interp_sBLSA_'+idvi+'_10_Processed_REST.nii.gz')
    if os.path.isfile(rsfile):
        rsfmri_list[jj] = rsfile
    else:
        raise FileNotFoundError(rsfile + " does not exist")

# add path to time series correlation maps to data table
ts_path = []
for idvi, rs_idvi in zip(idvi_list, rs_idvi_list):
    ts_fname = output_dir+'/output/time_series_correlation_maps/BLSA_'+idvi+'_rsBLSA_'+rs_idvi+'/remeaned_filreg_interp_sBLSA_'+rs_idvi+'_10_Processed_REST_reorient_fim_calc.nii'
    ts_path.append(ts_fname)
data_table['TSCM_nifti'] = ts_path
#data_table.to_csv('/cog/home/jacob/entorhinal-connectivity-and-tau/data/taurs_long_sample_VoxelStats.csv', index=False)

# MUSE labels for necessary ROIs
entorhinal_bilateral=[116,117]
entorhinal_rh=[116]
entorhinal_lh=[117]
hippocampus_bilateral=[47,48]
parahippocampalgyrus_bilateral=[170,171]
amygdala_bilateral=[31,32]
MTL_label = entorhinal_bilateral + hippocampus_bilateral + parahippocampalgyrus_bilateral + amygdala_bilateral

## nipype ##

infosource = Node(interface=IdentityInterface(fields=['idvi','rs_idvi']),
                  iterables=[('idvi', idvi_list),
                             ('rs_idvi', rs_idvi_list)],
                  synchronize=True, name="infosource")

av1451file = os.path.join('av1451_images','rBLSA_{idvi}_P2_av1451_c3_mean_rbv_pvc_suvr_mni.nii.gz')
rsfmri_file = os.path.join('rsfmri_images','BLSA_{rs_idvi}_10','remeaned_filreg_interp_sBLSA_{rs_idvi}_10_Processed_REST.nii.gz')
tsnr_file = os.path.join('rsfmri_images','BLSA_{rs_idvi}_10','bin_thr_tSNR_BLSA_{rs_idvi}_10.nii.gz')

templates = {'suvr_mni': av1451file,
             'rsfmri_images': rsfmri_file,
             'tSNR_images': tsnr_file}

selectfiles = Node(interface=nio.SelectFiles(templates, base_directory=images_dir), name="selectfiles")

def AlignOrient(in_file):
    import nibabel as nib
    from os.path import abspath
    from nipype.utils.filemanip import split_filename
    img = nib.load(in_file)
    template_file = '/cog/home/jacob/entorhinal-connectivity-and-tau/EPI.nii'
    template_img = nib.load(template_file)
    # reorient input image to match orientation of template image
    desiredOrient = nib.orientations.axcodes2ornt(nib.aff2axcodes(template_img.affine))
    currentOrient = nib.orientations.axcodes2ornt(nib.aff2axcodes(img.affine))
    orients = nib.orientations.ornt_transform(currentOrient,desiredOrient)
    reoriented_affine = img.affine.dot(nib.orientations.inv_ornt_aff(orients, img.shape))
    reoriented = nib.orientations.apply_orientation(img.get_data(),orients)
    # overwrite input image origin to match template
    reoriented_affine[:,-1] = template_img.affine[:,-1]
    # save reoriented image
    _, base, _ = split_filename(in_file)
    output_fname = abspath(base + '_reorient.nii.gz')
    nib.save(nib.Nifti1Image(reoriented, reoriented_affine), output_fname)
    return output_fname

reorientrs = Node(interface=Function(input_names=['in_file'],
                                   output_names=['out_file'],
                                   function = AlignOrient), name = "reorientrs")

reorienttsnr = Node(interface=Function(input_names=['in_file'],
                                   output_names=['out_file'],
                                   function = AlignOrient), name = "reorienttsnr")

# define array of zeros with same shape as tSNR maps to add images to
def tSNRtemplate():
    import numpy as np
    import nibabel as nib
    from os.path import abspath
    from nipype.utils.filemanip import split_filename
    rs_template = '/cog/home/jacob/entorhinal-connectivity-and-tau/EPI.nii'
    tsnr_img = nib.load(rs_template)
    tsnr_data = np.asanyarray(tsnr_img.dataobj)
    tsnr_zeros = np.zeros(tsnr_data.shape)
    _, base, _ = split_filename(rs_template)
    output_fname = abspath(base + '_zeros.nii.gz')
    tsnr_template = nib.save(nib.Nifti1Image(tsnr_zeros, tsnr_img.affine), output_fname)
    return output_fname

tsnrtemp = Node(interface=Function(input_names=[],
                                   output_names=['out_file'],
                                   function=tSNRtemplate), name = "tsnrtemp")

addtsnr = JoinNode(interface=fsl.maths.MultiImageMaths(op_string='-add %s '*len(rs_idvi_list)),
                                                       joinsource = "infosource", joinfield = "operand_files", name = "addtsnr")

threshtsnr = Node(interface=fsl.maths.Threshold(thresh=5,
                                                direction='below',
                                                args='-binv'), name = "threshtsnr")

mtlmask = Node(interface=MaskMUSEROI(in_file=muse_img,
                                     roi_list=MTL_label), name = "mtlmask")

mtlmaskreorient = Node(interface=fsl.ApplyXFM(reference=rs_template,
                                              interp='nearestneighbour',
                                              uses_qform=True), name = "mtlmaskreorient")

mtlmasktsnr = Node(interface=fsl.ApplyMask(nan2zeros=False,
                                           output_type='NIFTI'), name = "mtlmasktsnr")

subtractone = Node(interface=fsl.maths.BinaryMaths(operand_value=1,
                                                   operation='sub'), name = "subtractone")

images4d = JoinNode(interface=fsl.Merge(dimension='t'), joinsource = "infosource", joinfield = ["in_files"], name = "images4d")

onesamplettest = Node(interface=fsl.Randomise(one_sample_group_mean=True,
                                              num_perm=5000,
                                              mask=maskfile), name = "onesamplettest")

ecmask = Node(interface=MaskMUSEROI(in_file=muse_img,
                                    roi_list=entorhinal_lh), name = "ecmask")

maskTimage = Node(interface=fsl.maths.ApplyMask(nan2zeros=False), name = "maskTimage")

erodemask = Node(interface=fsl.maths.ErodeImage(kernel_shape='sphere', kernel_size=3), name = "erodemask")

reorientmask = Node(interface=fsl.ApplyXFM(reference=rs_template,
                                           interp='nearestneighbour',
                                           uses_qform=True), name = "reorientmask")

generateroi = Node(interface=ROIcreate(connectivity=2,
                                       size=1,
                                       roi_name='entorhinal_lh'), name = "generateroi")

maskseed_tsnrfreq = Node(interface=fsl.maths.ApplyMask(nan2zeros=False), name = "maskseed_tsnrfreq")

ecmasks2mni = Node(interface=fsl.ApplyXFM(reference=maskfile,
                                          interp='nearestneighbour',
                                          uses_qform=True), name = "ecmasks2mni")

# compute mean resting state signal within seed for each time series for each subject, output to text file
def idealfile(in_file, mask_file):
    import numpy as np
    import nibabel as nib
    from os.path import abspath
    from nipype.utils.filemanip import split_filename
    # read in resting state image
    rs_img = nib.load(in_file)
    rs_data = np.asanyarray(rs_img.dataobj)
    # read in seed image
    seed_img =  nib.load(mask_file)
    seed_data = np.asanyarray(seed_img.dataobj)
    # get mean of voxels inside the seed for each time point, save to list
    seedmeans = [np.mean(rs_data[:,:,:,tt][seed_data!=0]) for tt in range(0,rs_data.shape[3])]
    # save list of means to txt file
    _, base, _ = split_filename(in_file)
    output_fname = abspath(base + '_idealfile.txt')
    np.savetxt(output_fname, seedmeans, fmt='%.6f')

    return output_fname

idealfile = Node(interface=Function(input_names=['in_file','mask_file'],
                                    output_names=['out_txt_file'],
                                    function=idealfile), name = "idealfile")

selectmasks = Node(interface=IdentityInterface(fields=['roi_list','roi_name']),
                   iterables=[('roi_list', [MTL_label,entorhinal_bilateral,amygdala_bilateral,hippocampus_bilateral,parahippocampalgyrus_bilateral]),
                              ('roi_name', ['MTL','EC','AMY','HC','PHG'])],
                   synchronize=True, name = "selectmasks")

musemasks = Node(interface=MaskMUSEROI(in_file=muse_img), name = "musemasks")

musemasksreorient = Node(interface=fsl.ApplyXFM(reference=rs_template,
                                                interp='nearestneighbour',
                                                uses_qform=True), name = "musemasksreorient")

inverttsnr = Node(interface=fsl.UnaryMaths(operation='binv'), name = "inverttsnr")

musemaskstsnr = Node(interface=fsl.ApplyMask(nan2zeros=False), name = "musemaskstsnr")

def globconnval(in_files, mask_files, mask_name):
    import numpy as np
    import nibabel as nib
    import pandas as pd
    from os.path import abspath
    from nipype.utils.filemanip import split_filename
    globalconn_means = []
    for (img, mask) in zip(in_files, mask_files):
        rs_img = nib.load(img)
        rs_data = np.asanyarray(rs_img.dataobj)
        mask_img = nib.load(mask)
        mask_data = np.asanyarray(mask_img.dataobj)
        # data frame of resting state values for each voxel at each time point
        rs_df = pd.DataFrame([rs_data[:,:,:,tt][mask_data==1] for tt in range(0,rs_data.shape[3])])
        # compute correlation matrix between values for each voxel at each time point
        corrmatrix = rs_df.corr()
        #convert matrix to fischer-z scores and take mean of unqiue values (lower triangle)
        corrmatrixz = np.arctanh(corrmatrix)
        corrmatrix_lower = np.tril(corrmatrixz, -1)
        globalconn_means.append(np.mean(corrmatrix_lower[corrmatrix_lower!=0]))
    _, base, _ = split_filename(mask_name)
    output_fname = abspath(base + '_globconn.txt')
    #output_fname = 'global_connectivity_means_'+mask_name+'.txt'
    np.savetxt(output_fname, globalconn_means, fmt='%.6f')
    return output_fname

globconn = JoinNode(interface=Function(input_names=['in_files','mask_files','mask_name'],
                                       output_names=['out_txt_file'],
                                       function=globconnval), joinsource = "infosource", joinfield = ['in_files','mask_files'], name = "globconn")

# AFNI 3dfim+ to calculate the cross-correlation of an ideal reference waveform with the measured FMRI time series for each voxel
crosscorr = Node(interface=afni.Fim(out='Correlation',
                                    outputtype='NIFTI'), name = "crosscorr")

# AFNI 3dcalc to do fisher-z transformation voxelwise data
corrmap = Node(interface=afni.Calc(expr='atanh(a)',
                                   outputtype='NIFTI'), name = "corrmap")

# covarites for SPM multiple regression analysis
mreg_covariate_list = [{'vector':data_table['AV1451age_c'].tolist(),
                        'name':'age',
                        'centering':5},
                       {'vector':data_table['sex_c'].tolist(),
                        'name':'sex',
                        'centering':5},
                       {'vector':data_table['apoe4_c'].tolist(),
                        'name':'apoe',
                        'centering':5},
                       {'vector':data_table['entorhinal_c'].tolist(),
                        'name':'AV1451entorhinal',
                        'centering':5},
                       {'vector':data_table['fd_c'].tolist(),
                        'name':'motion',
                        'centering':5}]

mreg = JoinNode(interface=spm.MultipleRegressionDesign(covariates=mreg_covariate_list,
                                                       include_intercept=True,
                                                       global_normalization=1,
                                                       global_calc_omit=True,
                                                       threshold_mask_none=True,
                                                       use_implicit_threshold=True,
                                                       no_grand_mean_scaling=True,
                                                       use_mcr=True),
                                                       unique=True, joinsource='infosource', joinfield=['in_files'], name="mreg")

est = Node(interface=spm.EstimateModel(estimation_method={'Classical': 1},
                                       write_residuals=False,
                                       use_mcr=True), name="est")

estcon = Node(interface=spm.EstimateContrast(contrasts=[('agePos','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, 1, 0, 0, 0, 0]),
                                                        ('ageNeg','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, -1, 0, 0, 0, 0]),
                                                        ('sexMale','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, 0, 1, 0, 0, 0]),
                                                        ('sexFemale','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, 0, -1, 0, 0, 0]),
                                                        ('apoe4Pos','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, 0, 0, 1, 0, 0]),
                                                        ('apoe4Neg','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, 0, 0, -1, 0, 0]),
                                                        ('enttauPos','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, 0, 0, 0, 1, 0]),
                                                        ('enttauNeg','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, 0, 0, 0, -1, 0]),
                                                        ('motionPos','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, 0, 0, 0, 0, 1]),
                                                        ('motionNeg','T',['mean','age','sex','apoe','AV1451entorhinal','motion'],[0, 0, 0, 0, 0, -1])
                                                        ],
                                             group_contrast=True,
                                             use_mcr=True), name="estcon")

def glassbrain(in_file):
    from nilearn.plotting import plot_glass_brain
    from os.path import abspath
    from nipype.utils.filemanip import split_filename
    _, base, _ = split_filename(in_file)
    output_fname = abspath(base + '_glassbrain.png')
    plot_glass_brain(stat_map_img=in_file,
                     output_file=output_fname,
                     display_mode='lyrz',
                     colorbar=True,
                     axes=None,
                     #title=,
                     threshold=None,
                     annotate=True,
                     black_bg=False,
                     #cmap=,
                     alpha=0.5,
                     vmax=5,
                     plot_abs=False,
                     symmetric_cbar=True)
    return output_fname

threshold = MapNode(interface=spm.Threshold(contrast_index=1,
                                            extent_fdr_p_threshold=1,
                                            use_topo_fdr=False,
                                            use_fwe_correction=False,
                                            height_threshold=0.05,
                                            extent_threshold=10,
                                            use_mcr=True), iterfield = ['stat_image'], name='threshold')

thresh_nan = MapNode(interface=fsl.maths.MathsCommand(nan2zeros=True), iterfield = ['in_file'], name='thresh_nan')

thresh_glassbrain = MapNode(interface=Function(input_names=['in_file'],
                                     output_names=['out_png'],
                                     function=glassbrain), iterfield = ['in_file'], name = "thresh_glassbrain")

# set up output directory
datasink = Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
datasink.inputs.container = os.path.join('output')
datasink.inputs.substitutions = [('_idvi_','BLSA_'),
                                 ('_thresh','thresholded'),
                                 ('nan0','_agePos'),('nan1','_ageNeg'),('nan2','_sexMale'),('nan3','_sexFemale'),('nan4','_apoe4Pos'),
                                 ('nan5','_apoe4Neg'),('nan6','_enttauPos'),('nan7','_enttauNeg'),('nan8','_motionPos'),('nan9','_motionNeg'),
                                 ('_glassbrain0','agePos'),('_glassbrain1','ageNeg'),('_glassbrain2','sexMale'),('_glassbrain3','sexFemale'),
                                 ('_glassbrain4','apoe4Pos'),('_glassbrain5','apoe4Neg'),('_glassbrain6','enttauPos'),('_glassbrain7','enttauNeg'),
                                 ('_glassbrain8','motionPos'),('_glassbrain9','motionNeg'),
                                 ('_roi_list_',''),
                                 ('116.117.47.48.170.171.31.32_',''),('116.117_',''),('170.171_',''),('31.32_',''),('47.48_','')
                                 ]

# function to convert list outputs from Nodes to strings
def list_to_var(mylist):
    return mylist[0]

# generate one-sample t-test image of AV1451 images
onesampttest_wf = Workflow(name="onesampttest")
onesampttest_wf.connect([(subtractone, images4d, [('out_file','in_files')]),
                         (images4d, onesamplettest, [('merged_file','in_file')]),
                         (onesamplettest, maskTimage, [(('tstat_files', list_to_var),'in_file')]),
                         (ecmask, maskTimage, [('out_file','mask_file')])])
# generate mask of medial temporal lobe masked by summed thresholded tSNR image
tsnrmasks_wf = Workflow(name="tsnrmasks_wf")
tsnrmasks_wf.connect([(reorienttsnr, addtsnr, [('out_file','operand_files')]),
                      (tsnrtemp, addtsnr, [('out_file','in_file')]),
                      (addtsnr, threshtsnr, [('out_file','in_file')]),
                      (mtlmask, mtlmaskreorient, [('out_file','in_file')]),
                      (mtlmaskreorient, mtlmasktsnr, [('out_file','in_file')]),
                      (threshtsnr, mtlmasktsnr, [('out_file','mask_file')])])
# create ROI as seed for connectivity analysis
ecseeds_wf = Workflow(name="ecseeds_wf")
ecseeds_wf.connect([(erodemask, reorientmask, [('out_file','in_file')]),
                    (reorientmask, generateroi, [('out_file','in_file')]),
                    (generateroi, maskseed_tsnrfreq, [('out_file','in_file')])])
# calculate time series correlation maps using each seed
timeseriescorr_wf = Workflow(name="timeseriescorr_wf")
timeseriescorr_wf.connect([(reorientrs, idealfile, [('out_file','in_file')]),
                           (reorientrs, crosscorr, [('out_file','in_file')]),
                           (idealfile, crosscorr, [('out_txt_file','ideal_file')]),
                           (crosscorr, corrmap, [('out_file','in_file_a')])])
globalconnectivity_wf = Workflow(name="globalconnectivity_wf")
globalconnectivity_wf.connect([(selectmasks, musemasks, [('roi_list','roi_list')]),
                               (musemasks, musemasksreorient, [('out_file','in_file')]),
                               (musemasksreorient, musemaskstsnr, [('out_file','in_file')]),
                               (inverttsnr, musemaskstsnr, [('out_file','mask_file')]),
                               (musemaskstsnr, globconn, [('out_file','mask_files')]),
                               (selectmasks, globconn, [('roi_name','mask_name')]),
                               (reorientrs, globconn, [('out_file','in_files')])])
'''
# group-level analysis on masked time series correlation maps
grouplevelstats_wf = Workflow(name="grouplevelstats_wf")
grouplevelstats_wf.connect([(mreg, est, [('spm_mat_file','spm_mat_file')]),
                            (est, estcon, [('spm_mat_file','spm_mat_file'),
                                           ('residual_image','residual_image'),
                                           ('beta_images','beta_images')]),
                            (estcon, threshold, [('spm_mat_file','spm_mat_file'),
                                                 ('spmT_images','stat_image')]),
                            (threshold, thresh_nan, [('thresholded_map','in_file')]),
                            (thresh_nan, thresh_glassbrain, [('out_file','in_file')])])
'''
# combine individual workflows into nested workflow
analysis_workflow = Workflow(name="analysis_workflow")
analysis_workflow.base_dir = os.path.join(output_dir,'analysis_workflow_dir')
analysis_workflow.config = {"execution": {"crashdump_dir": os.path.join(output_dir,'crashdumps')}}
analysis_workflow.connect([(infosource, selectfiles, [('idvi','idvi'),
                                                      ('rs_idvi','rs_idvi')]),
                           (selectfiles, tsnrmasks_wf, [('tSNR_images','reorienttsnr.in_file')]),
                           (selectfiles, onesampttest_wf, [('suvr_mni','subtractone.in_file')]),
                           (onesampttest_wf, ecseeds_wf, [('maskTimage.out_file','erodemask.in_file')]),
                           (tsnrmasks_wf, datasink, [('addtsnr.out_file','tSNR_freqmap'), # sum of tSNR images in sample
                                                     ('mtlmasktsnr.out_file','MTL_masked_tSNR')]), # MTL masked by summed tSNR image
                           (tsnrmasks_wf, ecseeds_wf, [('threshtsnr.out_file','maskseed_tsnrfreq.mask_file')]),
                           (ecseeds_wf, datasink, [('generateroi.out_file','ecmask_funcspace')]), # entorhinal mask in native functional space
                           (selectfiles, timeseriescorr_wf, [('rsfmri_images','reorientrs.in_file')]),
                           (ecseeds_wf, timeseriescorr_wf, [('generateroi.out_file','idealfile.mask_file')]),
                           (tsnrmasks_wf, globalconnectivity_wf, [('reorienttsnr.out_file','inverttsnr.in_file')]),
                           (globalconnectivity_wf, datasink, [('globconn.out_txt_file','global_connectivity_means')]),
                           (timeseriescorr_wf, datasink, [('corrmap.out_file','time_series_correlation_maps')])]) # MTL time series correlation maps
'''
                           (timeseriescorr_wf, grouplevelstats_wf, [('corrmap.out_file','mreg.in_files')]),
                           (tsnrmasks_wf, grouplevelstats_wf, [('mtlmasktsnr.out_file','mreg.explicit_mask_file')]),
                           (grouplevelstats_wf, datasink, [('estcon.con_images','SPM_output'),
                                                           ('estcon.spmT_images','SPM_output.@T'),
                                                           ('estcon.spm_mat_file','SPM_output.@mat')]), # files for spm GUI visualization of results
                           (grouplevelstats_wf, datasink, [('thresh_nan.out_file','thresholded_images'), # thresholded T-value images to output
                                                           ('thresh_glassbrain.out_png','glassbrain_threshimages')])]) # glassbrain visualization of thresholded T maps
'''

analysis_workflow.write_graph('nested_workflow.dot', graph2use='colored', simple_form=True)

result = analysis_workflow.run('MultiProc', plugin_args={'n_procs': n_procs})
