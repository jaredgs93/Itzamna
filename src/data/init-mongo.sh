#!/bin/bash

# Espera a que MongoDB esté disponible
echo "Esperando a que MongoDB esté disponible..."
sleep 5

# Importa datos a la colección `rules_evaluation`
mongoimport --host localhost --db skills_evaluation --collection rules_evaluation \
    --file /docker-entrypoint-initdb.d/rules_evaluation.json --jsonArray --drop

# Importa datos a la colección `fuzzy_variables`
mongoimport --host localhost --db skills_evaluation --collection fuzzy_variables \
    --file /docker-entrypoint-initdb.d/fuzzy_variables.json --jsonArray --drop

echo "Datos importados correctamente."
