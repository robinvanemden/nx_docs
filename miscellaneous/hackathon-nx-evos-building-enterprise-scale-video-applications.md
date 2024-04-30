# Hackathon: Nx EVOS: Building Enterprise-Scale Video Applications

Welcome to the Nx hackerearth EVOS hackathon\![\
\
](https://nx.docs.scailable.net/)The NX AI team is working hard to prepare the NX AI platform for a release alongside NX Gen 6. The NX AI plugin will be packaged with the installation of NX Server Gen 6, but until then, some extra installation steps will be necessary while the NX AI platform is in beta.&#x20;

Follow the instructions closely to get NX and NX AI running on your system.

1. Install NX and create an account.\
   Follow the instructions in our docs to install Nx Mediaserver on your system. It is important to create an Nx Cloud account during this process and [connect your system](https://nx.docs.scailable.net/nx-ai-manager/1.-install-network-optix#connect-your-nx-meta-system-with-your-nx-cloud-account) to your Nx Cloud account: [Installation Instructions](https://nx.docs.scailable.net/nx-ai-manager/get-started-with-the-nx-ai-manager-plugin/1.-install-network-optix).
2. Log into NX AI Cloud with an NX account.\
   To register your device to the NX AI Cloud, you need to be logged in to the NX AI Cloud with the Nx account you created in the previous step. Kindly navigate to the NX AI Cloud and log in: [NX AI Cloud Manager](https://admin.sclbl.nxvms.com/login).
3. Install plugin.\
   Paste the following line in a terminal on your device:\
   `sudo bash -ic "$(wget -q -O - https://get.sclbl.net/nx_plugin)" package=Plugin-v4-0`\
   This process should install the plugin and prompt if it should restart the NX Mediaserver, it might be necessary to restart the NX Mediaserver for the plugin to be detected and properly loaded.\
   This step will no longer be necessary with the full release.
4. Continue with documentation\
   Consult the documentation about how to configure the plugin: [NX AI Plugin Configuration](https://nx.docs.scailable.net/nx-ai-manager/get-started-with-the-nx-ai-manager-plugin/2.-configure-the-nx-ai-manager-plugin)
5. Create and upload your own models\
   Create your own custom models and deploy them easily to the NX AI platform: [NX AI Models Introduction](https://nx.docs.scailable.net/for-data-scientists/introduction) - after which you can [upload your model](../nx-ai-cloud/upload-your-model/).
6. Create your own custom data processors.\
   Create your own custom data processors and easily integrate them with the NX AI Platform: [External Processors Docs.\
   ](https://nx.docs.scailable.net/nx-ai-manager/get-started-with-the-nx-ai-manager-plugin/7.-advanced-configuration/7.1-external-post-processing)Check out our examples and integration SDK to get started: [Integration SDK](https://github.com/scailable/sclbl-integration-sdk).
7. Have fun! Hack!
