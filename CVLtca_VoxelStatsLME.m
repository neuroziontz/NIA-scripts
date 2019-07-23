%%% CVLtca %%%

addpath(genpath('/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/VoxelStats'));
image_type = 'nifti';
string_model = 'CVLtca ~ AV1451age_c + sex + pibgroup + entorhinal_c + TSCM_nifti + interval + entorhinal_c:TSCM_nifti + TSCM_nifti:interval + entorhinal_c:interval + entorhinal_c:TSCM_nifti:interval + (1 + interval | blsaid)';
data_file = '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/taurs_long_sample_VoxelStats.csv';
mask_file = '/cog/home/jacob/entorhinal-connectivity-and-tau/nipype_output/output/MTL_masked_tSNR/atlas_blsa_muse_mni152_maskedROI_flirt_masked.nii';
multi_value_variables = {'TSCM_nifti'};
categorical_vars = {''};
include_string='';
[c_struct, slices_p, image_height_p, image_width_p, coeff_vars, voxel_num, df, voxel_dims] = VoxelStatsLME(image_type, string_model, data_file, mask_file, multi_value_variables, categorical_vars, include_string);

VoxelStatsWriteNifti(c_struct.tValues.Intercept, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_intercept_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.AV1451age_c, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_age_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.sex, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_sex_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.pibgroup, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_pibgroup_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.entorhinal_c, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_ectau_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.TSCM_nifti, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_TSC_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.interval, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_interval_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.entorhinal_cTSCM_nifti, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_ectaubyTSC_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.TSCM_niftiinterval, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_TSCbyinterval_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.entorhinal_cinterval, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_ectaubyinterval_Tmap.nii', mask_file);
VoxelStatsWriteNifti(c_struct.tValues.entorhinal_cTSCM_niftiinterval, '/cog/home/jacob/entorhinal-connectivity-and-tau/voxelstats_output/fitlme/CVLtca_ectaubyTSCbyinterval_Tmap.nii', mask_file);


VoxelStatsShowOnTemplate(c_struct.tValues.TSCM_niftiinterval, mask_file, 'nifti');
VoxelStatsShowOnTemplate(c_struct.tValues.entorhinal_cinterval, mask_file, 'nifti');
VoxelStatsShowOnTemplate(c_struct.tValues.entorhinal_cTSC_niftiinterval, mask_file, 'nifti');
