#!/bin/bash

mkdir -p data data/train data/test

urls="domain_urls.json"

download() {
    url=$1
    output=$2
    if wget -q --method=HEAD "$url"; then
        wget -O "$output" "$url"
        echo "Downloaded: $output"
    else
        echo "Failed to download: $url"
    fi
}

for domain in $(jq -r 'keys[]' "$urls"); do
    echo "Downloading domain: $domain"
    
    data_url=$(jq -r ".$domain.data" "$urls")
    download "$data_url" "data/${domain}.zip"

    unzip -q "data/${domain}.zip" -d "data"
    rm "data/${domain}.zip"

    train_url=$(jq -r ".$domain.train" "$urls")
    download "$train_url" "data/train/${domain}_train.txt"
    
    test_url=$(jq -r ".$domain.test" "$urls")
    download "$test_url" "data/test/${domain}_test.txt"
done