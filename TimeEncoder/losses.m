rutaImagen = 'C:\Users\paulo\Documents\GitHub\PUCV\TimeEncoder\PCA\orden.png';
imagen = imread(rutaImagen);

rutaPDF = 'C:\Users\paulo\Documents\GitHub\PUCV\TimeEncoder\PCA\orden';

figura = figure;
set(figura, 'Units', 'pixels', 'Position', [100, 100, size(imagen, 2), size(imagen, 1)]);

% Mostrar la imagen en la figura
imshow(imagen);

% Imprimir la figura en un archivo PDF
print(figura, rutaPDF, '-dpdf', '-r0');

% Cerrar la figura
close(figura);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

main_dir = 'C:\Users\paulo\Documents\GitHub\PUCV\TimeEncoder\PCA\Analysis\';
dictionary_f = {'ACTON275', 'GR2', 'BOL5', 'RX306', 'HALFAC3', 'IACCEL1', 'GR', 'ECE7'};

for i=dictionary_f
    train = strcat(main_dir, 'model_', i,'\t_encoder_train_losses.txt');
    test = strcat(main_dir, 'model_', i,'\t_encoder_test_losses.txt');

    fileID_train = fopen(train{1}, 'r');
    fileID_test = fopen(test{1}, 'r');

    data_train = textscan(fileID_train, '%s', 'delimiter', '\n');
    data_test= textscan(fileID_test, '%s', 'delimiter', '\n');

    fclose(fileID_train);
    fclose(fileID_test);

    data_train = data_train{1};
    data_test = data_test{1};

    L_train = [];
    L_test = [];
    epochs = [];

    for j = 1:length(data_train)
        L_train = horzcat(L_train,str2double(data_train{j}));
        L_test = horzcat(L_test,str2double(data_test{j}));
        epochs = horzcat(epochs,j);
    end

    filename = strcat('C:\Users\paulo\Documents\GitHub\PUCV\TimeEncoder\PCA\', i, '.pdf');
    figure;
    plot(epochs, L_train);
    hold on;
    plot(epochs, L_test);
    title(strcat('Señal',' ',i));
    legend('Train Loss', 'Test Loss')
    saveas(gcf, filename{1});
    clf;
    hold off;
    close all;
    filename = strcat(filename{1},',');
end