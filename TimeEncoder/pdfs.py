import PyPDF2
import os

# Lista de archivos PDF a concatenar
dir_pdfs = r'C:\Users\paulo\Documents\GitHub\PUCV\TimeEncoder'
# archivos = []
# for i in ['ACTON275', 'GR2', 'BOL5', 'RX306', 'HALFAC3', 'IACCEL1', 'GR', 'ECE7','orden']:
#     archivos.append(os.path.join(dir_pdfs,i+'.pdf'))

# Objeto para almacenar los archivos PDF combinados
pdf_combinado = PyPDF2.PdfMerger()

# Recorrer la lista de archivos y agregarlos al objeto PdfFileMerger
# for archivo in archivos:
#     pdf_combinado.append(archivo)
for archivo in os.listdir(r'C:\Users\paulo\Documents\GitHub\PUCV\TimeEncoder\data\results_test\pdfs'):
    pdf_combinado.append(os.path.join(r'C:\Users\paulo\Documents\GitHub\PUCV\TimeEncoder\data\results_test\pdfs',archivo))


# Guardar el archivo PDF combinado
pdf_combinado.write(os.path.join(dir_pdfs,'pdf_combinado.pdf'))

# Cerrar el objeto PdfFileMerger
pdf_combinado.close()
