function getPackageVersion() {
    local package_name="$1"

    local text=$(python -m pip show "$package_name")

    if [ $? -eq 1 ]; then
        exit 1;
    fi

    local error_match=$(echo "$text" | grep -i "^error")

    if [ $? -eq 1 ]; then
        exit 1;
    fi

    if [ -n "$error_match" ]; then
        exit 1;
    fi

    local version=$(echo "$text" | grep -oP 'Version: \K.*')

    if [ $? -eq 1 ] || [ -z $version ]; then
        exit 1;
    fi

    echo "$version"
}

function findPackageInRequirements() {
    local package_name="$1"

    local grep_output=$(grep -Enm1 "$package_name(==|\s|$)" requirements.txt)

    if [ $? -eq 1 ]; then
        echo
        exit 1;
    fi

    if [ -z "$grep_output" ]; then
        echo
        exit 1;
    fi

    local line_number=$(echo "$grep_output" | cut -f1 -d:)
    
    echo $line_number
}

function addPackageToRequirements() {
    local package_name="$1"
    local package_version="$2"

    echo "$package_name==$package_version" >> requirements.txt
}

function installPackage() {
    local package_name="$1"

    local line_number=$(findPackageInRequirements $package_name)

    if [ -n "$line_number" ]; then
        echo "[ERROR]: Package already registered on requirements.txt"
        exit 0;
    fi

    echo "[NOTICE]: Installing package $package_name"
    echo
    local installOutput=$(pip install "$package_name" --disable-pip-version-check | tee /dev/tty)
    echo

    if [ $? -eq 1 ]; then
        exit 1;
    fi

    local already_installed=$(echo "$installOutput" | grep -i "^requirement already satisfied: $package_name")

    if [ $? -eq 1 ]; then
        exit 1;
    fi

    local package_version=$(getPackageVersion $package_name)

    if [ -n "$already_installed" ]; then
        echo "[WARNING]: Package is already installed"
        echo "[NOTICE]: Adding package $package_name==$package_version to requirements.txt"
        addPackageToRequirements $package_name $package_version || true
        
        exit 0;
    fi

    echo "[NOTICE]: Adding package $package_name==$package_version to requirements.txt"
    addPackageToRequirements $package_name $package_version || true

    exit 0
}

"$@"
