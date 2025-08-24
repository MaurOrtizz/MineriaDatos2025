import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_file(
    dataset = 'shreyasur965/births-and-deaths',
    file_name = 'births-and-deaths-projected-to-2100.csv',
    path = './Practica 1',                                  #Estoy ejecutando el script desde el directorio de todas las practicas
    force=True)