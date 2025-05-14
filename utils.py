import os
import comtypes




# ========== CONVERT PPTX TO IMAGES ==========
def convert_pptx_to_images(pptx_path, output_folder):
    # Check if PowerPoint file exists
    if not os.path.exists(pptx_path):
        raise FileNotFoundError(f"PowerPoint file not found at: {pptx_path}\nPlease make sure the file exists and the path is correct.")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    try:
        powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
        powerpoint.Visible = 1
        presentation = powerpoint.Presentations.Open(pptx_path, WithWindow=False)
        presentation.SaveAs(os.path.abspath(output_folder), 17)  # 17 = ppSaveAsJPG
        presentation.Close()
        powerpoint.Quit()
        print(f"Successfully converted PowerPoint to images in {output_folder}")
    except Exception as e:
        print(f"PowerPoint conversion failed: {str(e)}")
        print("Please make sure:")
        print("1. Microsoft PowerPoint is installed on your system")
        print("2. The PowerPoint file is not corrupted")
        print("3. You have write permissions in the output folder")
        raise

