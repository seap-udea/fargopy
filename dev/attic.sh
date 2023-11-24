if [ $# -gt 0 ]
then
    for cmd in $@
    do
        if [ $cmd = "download" ]
        then
            echo "Installing..."
        fi
    done
fi
