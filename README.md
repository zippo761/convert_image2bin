# convert_image2bin
This module make rotation  and convert image to bin.
### Example:
import correct_img


path = r'C:/Anaconda3/envs/py33/file.pdf'

path_jpg = correct_img.spoil_page_pdf(path, 3)

output_image = correct_img.rotation(path_jpg)

output_image = correct_img.convert_img2bin(output_image)

correct_img.save_jpg(r'C:/Anaconda3/envs/py33/name_of_file.jpg', output_image) 
