// YAAIL - Yet Another Apple Image Loader
// Based off of Brad's YAIL for Atari
//

#include "fujinet-network.h"
#include "yail.h"

#define HGR_SCREEN_ADDRESS ((unsigned char*)0x2000)

char buffer[8192];
char result[1024];
char *version = "v1.3.14";
char *my_version = "v1.0.0";
char *url;
char* image_url = "n:http://fujinet.diller.org/APPLE/black.gr";
char* image_data;
uint8_t err = 0;
uint16_t conn_bw;
uint8_t connected;
uint8_t conn_err;
uint8_t trans_type = OPEN_TRANS_CRLF;

void debug() {}

void setup() {
    uint8_t init_r = 0;
    bzero(buffer, 8192);
    bzero(result, 1024);
    gotox(0);
    init_r = network_init();
    printf("init: %d, derr: %d\n", init_r, fn_device_error);
    #ifdef BUILD_APPLE2
    printf("nw: %d\n", sp_network);
    #endif
}

void handle_err(char *reason) {
    if (err) {
        printf("Error: %d (d: %d) %s\n", err, fn_device_error, reason);
        cgetc();
        exit(1);
    }
}

// Function to load the image from a remote URL
void load_image(const char* image_url, unsigned char* image_data) {
    url = (const char *)image_url;
    err = network_open(url, OPEN_MODE_HTTP_GET, 0);
    handle_err("open");
    err = network_read(url, buffer, 8192);
    handle_err("read");
    network_close(url);
    printf("simple read (same as GET):\n");
    handle_err("get:close");
}

// Function to display the image on the Apple II screen
void display_image(const unsigned char* image_data) {
    // Set the graphics mode to Hi-Res
    *(unsigned char*)0xC050 = 0;
    *(unsigned char*)0xC057 = 0;
    // Copy the image data to the Hi-Res screen buffer
    // Assuming the image size is 280x192 pixels (8192 bytes)
    memcpy(HGR_SCREEN_ADDRESS, image_data, 8192);
    // Wait for a key press before exiting
    cgetc();
}

int main() {
    setup();
    // Allocate memory for the image data
    // Assuming the image size is 280x192 pixels (8192 bytes)
    image_data = (char*)malloc(8192);
    printf("yaail %s\n", my_version);
    printf("fn-lib %s\n", version);
    printf("Base URL: %s\n", image_url);
    // Load the image from the remote URL
    load_image(image_url, image_data);
    // Display the image on the Apple II screen
    display_image(image_data);
    // Free the allocated memory
    free(image_data);
    return 0;
}