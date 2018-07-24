"""
Label4CV Architecture
Author: David Jung
Date: 2018-07-23 Mon
"""

if SELECT(image):
    if AUTO_LABEL_IS_READY:
        if not TAKE(RECOMMANDATION):
            data = (x, ground_truth) # Make a correction by hand
        else:
            # Slightly adjust the hyperparameters 
            # since user can make a mistake by chance
            data = (x, ground_truth, opt)
    else:
        data = (x, ground_truth) # Label the image by hand

    if data:
        buffer.append(data)
        if length(buffer) >= MIN_BATCH_SIZE:
            COLLECT(buffer) # Add batch to training pool
            FLUSH(buffer)
     
while POOL: # Train forever in backend
    TRAIN(POOL)
    if EVAL() >= THRESHOLD:
        EXPOSE(AUTO_LABEL) # Recommandation is ready
    elif EVAL() + BETA < THRESHOLD:
        CONCEAL(AUTO_LABEL) # Need improvement
