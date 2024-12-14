#!/bin/bash

# Wait for MongoDB to become available
echo "Wait for MongoDB to become available..."
sleep 5

# Import data into the collection `rules_evaluation`
mongoimport --host localhost --db skills_evaluation --collection rules_evaluation \
    --file /docker-entrypoint-initdb.d/rules_evaluation.json --jsonArray --drop

# Import data into the collection `fuzzy_variables`
mongoimport --host localhost --db skills_evaluation --collection fuzzy_variables \
    --file /docker-entrypoint-initdb.d/fuzzy_variables.json --jsonArray --drop

echo "Data imported correctly."
