bleu_base_url='https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/bleu'

bleu_files=("LICENSE" "__init__.py" "bleu.py" "bleu_scorer.py")



rouge_base_url="https://raw.githubusercontent.com/tylin/coco-caption/master/pycocoevalcap/rouge"

rouge_files=("__init__.py" "rouge.py")



download() {

    local metric=$1; shift;

    local base_url=$1; shift;

    local fnames=($@);



    mkdir -p ${metric}

    for fname in ${fnames[@]};

    do

        printf "downloading: %s\n" ${base_url}/${fname}

        wget --no-check-certificate ${base_url}/${fname} -O ${metric}/${fname}

    done

}



# prepare rouge

download "rouge_metric" ${rouge_base_url} ${rouge_files[@]}



# prepare bleu

download "bleu_metric" ${bleu_base_url} ${bleu_files[@]}
