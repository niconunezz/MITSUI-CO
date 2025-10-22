# MITSUI-CO

# Guia de pull

### No tocar sin avisar explicitamente
- data_loader.py
- data_proccesing.py
- carpetas de modelos
- tensors
- train.py (a menos de que sea necesario)

### Workflow para contribuir
- crear carpeta para tu modelo
- el modelo debe seguir el diseño tipico de torch (ver Ilya)
- crear una config en cofigs.py especifica para parametros de tu modelo (se puede hacer mas de 1) y añadirla a Configs.
- hacerlo suficientemente general para que corra train.py
