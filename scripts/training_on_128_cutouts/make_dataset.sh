#!/bin/bash
#SBATCH --account=def-lplevass
#SBATCH --mem=8G                             # memory per node
#SBATCH --time=00-15:00
source $HOME/environments/scope/bin/activate
python $DEBLENDER/scripts/skirt_fits_to_h5.py\
        --size=512\
        --downsample=2\
        --filters SUBARU_HSC.G SUBARU_HSC.R SUBARU_HSC.I SUBARU_HSC.Z SUBARU_HSC.Y\
        --skirt_path=$HOME/projects/rrg-lplevass/data/SKIRT_TNG\
        --output_path=$HOME/scratch/skirt128_grizy_microjy.h5
