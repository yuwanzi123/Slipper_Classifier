for file in *.jpg; do
    new_file="$(mktemp XXXXXXXX.jpg)"
    mv -f -- "$file" "$new_file"
done
