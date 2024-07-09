---
description: Things to check if the plugin is running correctly, and how to check it.
---

# Plugin checks

First make sure the [system checks](system-checks.md) are all correct.

## Check the plugin installation

You can check on the device settings page if the "Nx AI Manager" plugin is present in the plugins tab. If the plugin tab is not present or if the "Nx AI Manager" is not there you need to download the plugin.

{% hint style="warning" %}
Ensure you download Nx Meta version 6.0 or later to be able to use the Nx AI Plugin.
{% endhint %}

### Is the plugin available

When opening the camera settings the Plugins tab is available, and on the Plugins tab the Nx AI Manager is present

<figure><img src="../../.gitbook/assets/Screenshot from 2024-07-03 10-58-41.png" alt=""><figcaption></figcaption></figure>

### Is the plugin installed

If the plugin is not available, check if the plugin is present (and what version it is)

* You need Nx Meta version 6.0 or newer.
* The plugin version needs to be version 4.0 or higher. This document uses the plugin version 4.1 or higher.
  *   It could be the case that the plugin is not properly installed. Follow the instructions at [install-nx-ai-manager-plugin.md](../../nx-ai-manager/install-nx-ai-manager-plugin.md "mention"), and check that the plugin file is installed at:

      `/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/libnxai_plugin.so`
  * If the plugin file is not there, or it exists but the server still cannot detect it, follow the instructions at [manual-plugin-installation.md](../../nx-ai-manager/manual-plugin-installation.md "mention")

## Are the needed drivers installed

### NVIDIA Orin accelerated

If you're running on NVIDIA Orin - is a current version of NVIDIA Jetpack installed?&#x20;

### Hailo Accelerated Devices

In order for the runtime using Hailo accelerators to work, the correct Hailo Runtime (**hailort)**  needs to be installed. Currently, the Nx AI Runtime supports Hailo driver version **4.17.0**. Ensure that the correct runtime is installed.

## Is the Nx AI Runtime running

### Check the Nx EVOS plugin interface

If the plugin is not running the Plugin tab will show a deactivated toggle for the device.

<figure><img src="../../.gitbook/assets/Screenshot from 2024-07-03 10-58-58.png" alt=""><figcaption></figcaption></figure>

You can enable the NX AI Runtime by clicking the Device active switch.

<figure><img src="../../.gitbook/assets/Screenshot from 2024-07-03 10-59-01.png" alt=""><figcaption></figcaption></figure>

### Check the command line

Check the console if it is running with the following command:

```shell
ps axu | grep -E -i -w "sclbld|sclblmod"
```

The return should be something similar to if the plugin is running:

```shell
user@system:~$ ps axu | grep -E -i -w "sclbld|sclblmod"
network+  797274 43.9  0.4  44188 33404 ?        Sl   12:14   8:54 /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/bin/sclblmod
network+  797295  674  1.8 740244 143308 ?       Sl   12:14 136:34 /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/bin/../bin/sclbld [{"ModelPath": "/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/bin/../cache/7b65bdda-39da-4259-b1bf-b0d1dbb7b162.onnx", "RuntimePath": "/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/bin/../bin/libRuntimeLibrary.so"}] /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/bin/../sockets/sclblmod_to_sclbld_0 /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/bin/../sockets/sclbld_to_sclblmod_0 1 64
```

If none of these lines are present, the plugin is not running.

## Is it a model problem

First make sure that the model actually has something to detect. For example if you're using a model that is expecting vehicles, that will not be detected on a camera that is viewing an empty warehouse.

### Is the model working

Check if the module is started by looking into the logfile at `/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/etc/sclblmod_log.log`.

```bash
tail -f /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/etc/sclblmod_log.log
```

It should show updates like the following:

```
MODULE: 1720434595502 000000070: Notice: Model run  3727 
MODULE: 1720434595502 000000064: Notice: Postprocesssing [1]
MODULE: 1720434595502 000000104: Notice: Converting bboxes to image space 
MODULE: 1720434595503 000000193: Notice: Could not find bboxes array in inference results.
MODULE: 1720434595503 000000077: Notice: Could not find scores output in inference results.
MODULE: 1720434595503 000000070: Inference completed with BBoxes: 3, Scores: 0, Counts: 0
```

If there is no output, or the only output is an error message the plugin is not running.

### Is the model downloaded

Check if the model file exists and has the correct size

See if the model with the ID from the cloud is downloaded and has the correct size.&#x20;

For example the **80-Classes Object Detector \[640x640]** model has the following model ID: `7b65bdda-39da-4259-b1bf-b0d1dbb7b162` on the model detail page:

<figure><img src="../../.gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

Models are downloaded into `/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/cache/` on the local file system.&#x20;

Open a SSH session to the server and use the `ls` command to list the model file and verify if it matches the expected size, you can compare the files by manually downloading the model file and checking it with the version in the cache on the system:

```bash
ls --full-time -h /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/cache/7b65bdda-39da-4259-b1bf-b0d1dbb7b162.*
-rwxrwxrwx 1 networkoptix-metavms networkoptix-metavms 24M 2024-05-22 12:40:39.151613981 +0200 /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/cache/7b65bdda-39da-4259-b1bf-b0d1dbb7b162.onnx
```





