#include <stdio.h>
#include <stdlib.h>
#include <apple2.h>
#include <conio.h>
#include <string.h>

// Function prototypes
void init_text_mode(void);
void display_menu(void);
void init_graphics(void);
void load_local_image(const char *filename);
void load_remote_image(const char *image_url, unsigned char *image_data);
void display_image(void);
void handle_err(const char *msg);

unsigned char *image_buffer;

int main() {
    char choice;

    // Initialize text mode and display menu
    init_text_mode();
    display_menu();

    // Wait for user input
    choice = cgetc();

    if (choice == 'L' || choice == 'l') {
        // Initialize graphics
        init_graphics();

        // Load the local test image
        load_local_image("test.hrg");

        // Display the image
        display_image();
    } else if (choice == 'F' || choice == 'f') {
        // Initialize graphics
        init_graphics();

        // Allocate memory for the image
        image_buffer = (unsigned char *)malloc(8192);
        if (!image_buffer) {
            printf("Error: Could not allocate memory for image\n");
            exit(1);
        }

        // Load the remote image
        load_remote_image("http://example.com/test.hrg", image_buffer);

        // Display the image
        display_image();
    } else {
        printf("Invalid choice.\n");
    }

    // Wait for a key press before exiting
    while (!kbhit()) {}

    // Free the allocated image buffer
    if (image_buffer) {
        free(image_buffer);
    }

    return 0;
}

void init_text_mode() {
    // Set the text mode
    text();
    home();
    clrscr();
}

void display_menu() {
    // Print the menu
    cputs("Welcome to Yet Another Image Loader for Apple\n");
    cputs("L - Press to load local HRG image\n");
    cputs("F - Press to load remote HRG with FujiNet\n");
}

void init_graphics() {
    // Set the graphics mode to HiRes
    hires();
}

void load_local_image(const char *filename) {
    FILE *file;
    long file_size;

    // Open the file
    file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    // Get the file size
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for the image
    image_buffer = (unsigned char *)malloc(file_size);
    if (!image_buffer) {
        printf("Error: Could not allocate memory for image\n");
        fclose(file);
        exit(1);
    }

    // Read the file into the buffer
    fread(image_buffer, 1, file_size, file);

    // Close the file
    fclose(file);
}

void load_remote_image(const char *image_url, unsigned char *image_data) {
    const char *url;
    int err;

    // Placeholder function to simulate loading an image from a remote URL
    // This function will need to be replaced with actual FujiNet network code

    url = (const char *)image_url;
    err = network_open(url, OPEN_MODE_HTTP_GET, 0);
    handle_err("open");
    err = network_read(url, image_data, 8192);
    handle_err("read");
    network_close(url);
    printf("Image loaded from URL: %s\n", image_url);
}

void display_image() {
    // Copy the image buffer to the graphics screen
    unsigned char *screen = (unsigned char *)0x2000;
    for (int i = 0; i < 8192; i++) {
        screen[i] = image_buffer[i];
    }
}

void handle_err(const char *msg) {
    // Placeholder function to handle network errors
    printf("Error: %s\n", msg);
    exit(1);
}
