## Organzing all Uv-Vis data till 2026-03-12

date = [
    '20230503_CsPbBr_ZnI2', 
    '20230525_CsPbBr_ZnI_6mM', 
    '20230526_CsPbBr_ZnCl_6mM', 
    '20230612_CsPbBr_ZnI', 
    '20230726_CsPbBr_ZnI', 
    '20230925_insitu_dilute', 
    '20231030_kafka_ML', 
    '20231031_kafka_ML', 
    '20231103_OA_color', 
    '20231128_kafka_ML', 
    '20231201_iterate_ML', 
    '20231201_kafka_ML', 
    '20231204_iterate_ML', 
    '20231205_iterate_ML', 
    '20231206_iterate_ML', 
    '20231207_iterate_ML', 
    '20231208_iterate_ML', 
    '20231211_iterate_ML', 
    '20231212_iterate_ML', 
    '20231214_iterate_ML', 
    '20231215_iterate_ML', 
    '20231218_iterate_ML', 
    '20231221_iterate_ML', 
    '20231227_iterate_ML', 
    '20231228_iterate_ML', 
    '20240102_iterate_ML', 
    '20240104_iterate_ML', 
    '20240105_single_ZnI', 
    '20240108_iterate_ML', 
    '20240109_iterate_ML', 
    '20240110_iterate_ML', 
    '20240111_revise_model',    ## looks like assembled one
    '20240117_all_halides', 
    '20240118_all_halides', 
    '20240118_check_ZnI',       ## ZnI assembled
    '20240119_all_halides', 
    '20240122_all_halides', 
    '20240123_all_halides', 
    '20240124_all_halides', 
    '20240125_all_halides', 
    '20240129_ZnI2_60mM', 
    '20240130_ZnI2_60mM', 
    '20240201_all_halides', 
    '20240202_all_halides', 
    '20240205_all_halides', 
    '20240206_all_halides', 
    '20240207_all_halides', 
    '20240214_post_dilute', 
    '20240215_post_dilute', 
    '20240222_NSLSII_video', 
    '20240304_iterate_ML', 
    '20240304_kafka_single',    ## looks like assembled one
    '20240305_iterate_ML', 
    '20240307_kafka_single',    ## only assembled csv files
    '20240311_all_halides', 
    '20240312_kafka_single',    ## looks like assembled one
    '20240408-12_publisher',    ## looks like assembled one
    '20240408_post_dilute', 
    '20240410_post_dilute', 
    '20240412_post_dilute', 
    '20240417-19_publisher',    ## looks like assembled one
    '20240417_post_dilute', 
    '20240418_post_dilute', 
    '20240419_post_dilute', 
    '20240423_post_dilute', 
    '20240424_post_dilute', 
    '20240425_post_dilute', 
    '20240426_post_dilute', 
    '20240429_post_dilute', 
    '20240430_post_dilute', 
    '20240501_post_dilute', 
    '20240507_post_dilute', 
    '20240508_post_dilute', 
    '20240514_post_dilute',
    '20240516_post_dilute', 
    '20240520_post_dilute', 
    '20240521_post_dilute', 
    '20240715_XPD', 
    '20240717_PS_video',        ## looks like assembled one
    '20240812_v2_test', 
    '20240820_macro_Br', 
    '20240821_macro_Br', 
    '20240822_macro_Cl', 
    '20240826_macro_Cl', 
    '20240829_macro_Cl', 
    '20240902_macro_Cl', 
    '20250603_Cs_Pb-rich', 


]


# Add a simple plan
def count_test(detectors, *, num=1, delay=1):
    yield from count(detectors, num=num, delay=delay)

# Wait for some time to emulate the script with longer execution time
ttime.sleep(30)

[
'Cs_033_Br_167_ZnCl_100_OAm_000_Tol_1500', 
'Cs_033_Br_167_ZnCl_100_OAm_010_Tol_1550', 
'Cs_033_Br_167_ZnCl_100_OAm_020_Tol_1600', 
'Cs_033_Br_167_ZnCl_100_OAm_030_Tol_1650', 
'Cs_033_Br_167_ZnCl_100_OAm_040_Tol_1700', 
'Cs_033_Br_167_ZnCl_140_OAm_000_Tol_1700', 
'Cs_033_Br_167_ZnCl_130_OAm_010_Tol_1700', 
'Cs_033_Br_167_ZnCl_120_OAm_020_Tol_1700', 
'Cs_033_Br_167_ZnCl_110_OAm_030_Tol_1700']



[
'Cs_033_Br_167_ZnCl_100_OAm_000_Tol_1500', 
'Cs_033_Br_167_ZnCl_100_OAm_010_Tol_1550', 
'Cs_033_Br_167_ZnCl_100_OAm_020_Tol_1600', 
'Cs_033_Br_167_ZnCl_100_OAm_030_Tol_1650', 
'Cs_033_Br_167_ZnCl_100_OAm_040_Tol_1700', 
'Cs_033_Br_167_ZnCl_110_OAm_030_Tol_1700', 
'Cs_033_Br_167_ZnCl_120_OAm_020_Tol_1700', 
'Cs_033_Br_167_ZnCl_130_OAm_010_Tol_1700', 
'Cs_033_Br_167_ZnCl_140_OAm_000_Tol_1700'
]
