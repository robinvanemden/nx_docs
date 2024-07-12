# 6. Removing the Nx AI Manager

## Uninstalling the Nx AI Manager

It might be desired to remove the Nx AI Manager installation, due to a corrupted install or if you want to reinstall the Nx AI Manager with a different acceleration library.

The AI Manager can be removed by simply deleting the folder, assuming default installation, log in to the server console and execute the following command:

`sudo rm -r /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/`

When the mediaserver is restarted, the plugin will reinstall the Nx AI Manager.

## Removing the Nx AI plugin

To remove the Nx AI plugin from your Nx Meta server, log in to the server console and execute the following command:

`sudo rm -r /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin`

## Restarting the Nx Meta mediaserver

After uninstalling the Nx plugin and AI manager, restart the Nx mediaserver for these changes to take effect with the following command:

`sudo service networkoptix-metavms-mediaserver restart`

Please refer to Nx documentation for Nx Meta mediaserver uninstallation instructions
