# Reinforcement Learning - Snake Environment

![alt text](https://github.com/alexanderplatas/RL-snake-env/blob/master/game_image.png?raw=true)

## Ficheros

### Environments
- snakeenv.py: Entorno Snake con observaciones por defecto.
- snakeenv2.py: Entorno Snake con observaciones modificadas.

### Trained models (path models)
- ppo_snake_prueba2.model (ID = 2)
- ppo_snake_prueba4.model (ID = 4)
- ppo_snake_obs2.model    (ID = 5)

### Scripts

- checkenv.py: Script para verificar la implementación del entorno.
- train_model.py: Script para entrenar modelos nuevos.
- test_model.py: Script para evaluar modelos.

    Forma de uso:

```
python3 test_model.py <ID>
```


### Other files
- common.py: Contiene el método evaluate_policy()
