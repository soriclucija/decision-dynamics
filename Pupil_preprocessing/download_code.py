import os
import shutil

# Paths
source_root = r"J:\Workgroups\FSW\VISUAL-DECISIONS\subjects_2024"
asc_dest = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\raw_eyelink"
hdf_dest = r"C:\Users\lucij\Desktop\Leiden\Year 2\Thesis Project\2024_data\processed_eyelink"

# Make sure destination folders exist
os.makedirs(asc_dest, exist_ok=True)
os.makedirs(hdf_dest, exist_ok=True)

# Loop through each subject folder
for subj in os.listdir(source_root):
    subj_path = os.path.join(source_root, subj)

    if os.path.isdir(subj_path) and subj.isdigit() and int(subj) >= 20:
        
        raw_folder = os.path.join(subj_path, "raw_eyelink_data")
        if os.path.exists(raw_folder):
            asc_files = [f for f in os.listdir(raw_folder) if f.endswith(".asc")]
            if asc_files:
                asc_file = asc_files[0]  
                src_asc = os.path.join(raw_folder, asc_file)
                dst_asc = os.path.join(asc_dest, asc_file)
                shutil.copy2(src_asc, dst_asc)
                print(f"Copied ASC: {subj}/{asc_file}")
            else:
                print(f"No ASC file found for subject {subj}")
        else:
            print(f"No raw_eyelink_data folder for subject {subj}")

        alf_folder = os.path.join(subj_path, "alf")
        if os.path.exists(alf_folder):
            src_hdf = os.path.join(alf_folder, "processed_pupil.hdf")
            if os.path.exists(src_hdf):
                dst_hdf = os.path.join(hdf_dest, f"{subj}_processed_pupil.hdf")
                shutil.copy2(src_hdf, dst_hdf)
                print(f"Copied HDF: {subj}_processed_pupil.hdf")
            else:
                print(f"No HDF file found for subject {subj}")
        else:
            print(f"No alf folder for subject {subj}")

print("All done (starting from subject 020)!")

