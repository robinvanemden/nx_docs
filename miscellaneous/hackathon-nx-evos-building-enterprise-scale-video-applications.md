# Hackathon: Nx EVOS: Building Enterprise-Scale Video Applications

Welcome to the Nx hackerearth EVOS hackathon\![\
\
](https://nx.docs.scailable.net/)The NX AI team is working hard to prepare the NX AI platform for a release alongside NX Gen 6. The NX AI plugin will be packaged with the installation of NX Server Gen 6, but until then, some extra installation steps will be necessary while the NX AI platform is in beta. Some steps will be deprecated and unnecessary with the full release and will be clearly indicated.

Follow the instructions closely to get NX and NX AI running on your system.

1. Install NX and create an account.\
   Follow the instructions in our docs to install Nx Mediaserver on your system. It is important to create an Nx Cloud account during this process: [Installation Instructions](https://nx.docs.scailable.net/nx-ai-manager/get-started-with-the-nx-ai-manager-plugin/1.-install-network-optix).
2. Log into NX AI Cloud with an NX account.\
   To register your device to the NX AI Cloud, you need to be logged in to the NX AI Cloud with the Nx account you created in the previous step. Kindly navigate to the NX AI Cloud and log in: [NX AI Cloud Manager.\
   ](https://admin.sclbl.nxvms.com/nx-login)This step will no longer be necessary with the full release.
3. Install NX AI Manager:\
   The NX AI platform comes in two parts. First, a plugin that integrates with the NX Mediaserver, followed by the NX AI Runtime, which runs as a separate process. First, install the NX AI Runtime by pasting the following line into a terminal on your device:\
   `sudo bash -ic "$(wget -q -O -` [`https://get.sclbl.net/hackathon/manager.html`](https://get.sclbl.net/hackathon/manager.html)`)"`\
   Follow the prompted instructions to complete the installation.\
   This step will no longer be necessary with the full release.
4. Register NX AI Manager\
   Once the NX AI Runtime is installed, the device needs to be registered with the NX AI Cloud so that you can add/assign models.\
   Navigate to [http://localhost:8081/](http://localhost:8081/) , you will be prompted to register.\
   \
   ![](https://lh7-us.googleusercontent.com/v2EHq7gCfYNIhWlVT61ASGbMiZIGgFlaf2iDwrYikYWxI-F6c5B\_cjM5EOAr9zD6jcAputUWyXLeX-BDJIf1sTPWcop4HyVa9\_VZedmnH-rnpnaU66ajEMlNIjWvhHOdrY96ncgF9KQkzpfhWpamUPo)
5. Click on the ‘Register’ button to start the registration process.\
   ![](https://lh7-us.googleusercontent.com/mIAgsPomKmMjHXpAARxgUFvL-JCr9eix2DAkcZpVVVBgEj9Ah5tqNFft1B8F8naVdEwjUXcuqwjmG-VWw-AcQQ3Mhcky6uJT07xc3LQ930mM2m6zrLXxgUPejelMns-ysKGRiVDzAzuH2isUN3Gw\_WE)\
   If you are prompted for login information, then it’s possible you were not previously logged in with your NX account. Close the tab, redo step 2, and then attempt this step again.\
   The IP address might be different if you are not on the device where the NX AI Runtime is installed. The installation script of the previous step should log some possible IP addresses you can use to access the device’s Web UI and register.\
   This step will no longer be necessary with the full release.
6. Install plugin\
   Once the NX AI runtime is installed and running, it’s time to install the plugin. Paste the following line in a terminal on your device:\
   `sudo bash -c "$(wget -q -O -` [`https://get.sclbl.net/hackathon`](https://get.sclbl.net/hackathon)`)"  package=default`\
   This process should install the plugin and prompt if it should restart the NX Mediaserver, it might be necessary to restart the NX Mediaserver for the plugin to be detected and properly loaded.\
   This step will no longer be necessary with the full release.
7. Continue with documentation\
   Consult the documentation about how to configure the plugin: [NX AI Plugin Configuration](https://nx.docs.scailable.net/nx-ai-manager/get-started-with-the-nx-ai-manager-plugin/2.-configure-the-nx-ai-manager-plugin)
8. Create and upload your own models\
   Create your own custom models and deploy them easily to the NX AI platform: [NX AI Models Introduction](https://nx.docs.scailable.net/for-data-scientists/introduction) - after which you can [upload your model](../nx-ai-cloud/upload-your-model.md).
9. Create your own custom data processors.\
   Create your own custom data processors and easily integrate them with the NX AI Platform: [External Processors Docs.\
   ](https://nx.docs.scailable.net/nx-ai-manager/get-started-with-the-nx-ai-manager-plugin/7.-advanced-configuration/7.1-external-post-processing)Check out our examples and integration SDK to get started: [Integration SDK](https://github.com/scailable/sclbl-integration-sdk).
10. Have fun! Hack!
