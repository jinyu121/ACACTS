# KNOWN ISSUES

There are some known issues.

PLEASE HELP ME TO WORK THEM OUT! THANK YOU!

1. The crawler can not stop by itself.
    > You can stop it by yourself.

1. I do not konw if the CIEL\*a\*b\* regularization is right.
    > I found that different tool do the function `rgb2lab` in different ways, and get different results. We want to regularize each channel to $[0,1]$.

1. Luminance transfer do not work well, as well as face luminance correct.
    > Yes. We do not know where was wrong. PLEASE help us correct it!
    > We add a gaussian-like mask on faces as a dirty fix.

1. Should I devide the semantic-style score by the items counting in the semantic class?
