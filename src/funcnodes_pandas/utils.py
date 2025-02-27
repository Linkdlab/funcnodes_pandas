import re


def to_valid_identifier(
    s,
    max_length=30,
    digit_prefix="_",
    drop_characters="!\"#$%&'()*+,./:;<=>?@[\\]^`{|}~",
    merge_underscores=True,
):
    # remove unwanted characters
    s = s.translate(str.maketrans("", "", drop_characters))

    # Remove any characters that are not alphanumeric or underscores
    s = re.sub(r"\W|^(?=\d)", "_", s)

    # Ensure the identifier doesn't start with a digit
    if s[0].isdigit():
        s = digit_prefix + s

    # Merge consecutive underscores
    if merge_underscores:
        s = re.sub("_+", "_", s)

    # Truncate the identifier to the maximum length
    s = s[:max_length]

    return s
