# Normalization

Think of a digital image as a big grid filled with tiny colored dots, which we call "pixels." Each pixel contains colors—usually a mix of red, green, and blue. These colors have values that range from 0 to 255. A value of 0 means there is none of that color in the pixel (it's totally off), and a value of 255 means that color is shining as brightly as possible.

Now, suppose we want to make it easier for a computer to analyze and compare different images. One way to do this is by "normalizing" the color values in the image. Normalization is just a fancy term for adjusting these values so they fit within a new, consistent range, which helps in comparing images more fairly.

Here's how the normalization formula works:&#x20;

* **normalized\_color\_value = (original\_color\_value - mean) / scale;**

In this formula:

* **original\_color\_value** is the initial value of the color (anywhere from 0 to 255).
* **mean** is the average of all the color values. Subtracting this mean helps center our color values around zero.
* **scale** is a number we divide by to keep the values within a new, smaller range. This could be something like the largest color difference or another predefined number.

For example:

* If the average (mean) color value is 100, and our scale is 50:
  * For a pixel with a red color value of 150:
    * We subtract the mean: $$150−100=50$$
    * Then we divide by the scale: $$50 / 50​=1$$
  * So, the normalized red value would be 1.

This process transforms the original color values to a new scale that's easier for the computer to work with, typically ranging between -1 and 1 or 0 and 1. It's like changing the measurements of something from a variety of units (inches, centimeters, yards) all into meters so that they are easier to compare.

In the NX AI Cloud, you can set the normalization values as an integer array. For instance, for an RGB image, you might use \[123,234,242] for means and \[100,232,33] for scales. If you do not want to transform the **original\_color\_value**, just use 0 for mean, and 1 for scale— that is, \[0,0,0] and \[1,1,1] for an RGB input.
