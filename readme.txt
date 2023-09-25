source envs/code_exercise_api310/bin/activate
gunicorn -w 4 -k uvicorn.workers.UvicornWorker code_exercise_api:app --bind 0.0.0.0:2112


