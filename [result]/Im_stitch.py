#%%
from PIL import Image

def merge_image2(file1, file2):
    #Merge two images into one, displayed side by side
    #:param file1: path to first image file
    #:param file2: path to second image file
    #:return: the merged Image object
 
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = max(width1,width2)
    result_height = height1 + height2

    result = Image.new('RGB', (result_width, result_height),color = 'white')
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))
    return result

def side_merge_image2(file1, file2):
    #Merge two images into one, displayed side by side
    #:param file1: path to first image file
    #:param file2: path to second image file
    #:return: the merged Image object
 
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1,height2)

    result = Image.new('RGB', (result_width, result_height),color = 'white')
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result

def merge_image3(file1, file2,file3):
	image1 = Image.open(file1)
	image2 = Image.open(file2)
	image3 = Image.open(file3)
	
	(width1, height1) = image1.size
	(width2, height2) = image2.size
	(width3, height3) = image3.size
	
	result_width = max(width1,width2,width3)
	result_height = height1 + height2 + height3
	
	result = Image.new('RGB', (result_width, result_height),color = 'white')
	result.paste(im=image1, box=(0, 0))
	result.paste(im=image2, box=(0, height1))
	result.paste(im=image3, box=(0, height1 + height2))
	return result


merge_image3(
    "DPOvsTrueValue_full_(A).jpeg",
    "DPOvsTrueValue_full_(B).jpeg",
    "DPOvsTrueValue_full_(C).jpeg"
).save("DPOvsTrueValue_full.jpeg")

merge_image3(
    "DPOvsTrueValue_trun_(A).jpeg",
    "DPOvsTrueValue_trun_(B).jpeg",
    "DPOvsTrueValue_trun_(C).jpeg"
).save("DPOvsTrueValue_trun.jpeg")

merge_image3(
    "true_vs_pred_full(A).jpeg",
    "true_vs_pred_full(B).jpeg",
    "true_vs_pred_full(C).jpeg"
).save("true_vs_pred_full.jpeg")

merge_image3(
    "true_vs_pred_trun(A).jpeg",
    "true_vs_pred_trun(B).jpeg",
    "true_vs_pred_trun(C).jpeg"
).save("true_vs_pred_trun.jpeg")

side_merge_image2(
    "true_vs_pred_full.jpeg","true_vs_pred_trun.jpeg"
).save("true_vs_pred.jpeg")

merge_image3(
    "RepeatRMSE_BG_.jpeg",
    "RepeatRMSE_EA_.jpeg",
    "RepeatRMSE_IP_.jpeg"
).save("RepeatRMSE.jpeg")

merge_image2(
    "RepeatParams_full.jpeg",
    "RepeatParams_trun.jpeg"
).save("RepeatParams.jpeg")

# %%
