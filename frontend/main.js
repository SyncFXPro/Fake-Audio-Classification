console.log('FAC application loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded');

});

document.getElementById('submit-button').addEventListener('click', function() {
    console.log('Submit button clicked');
});

// Upload mp3 file to the server

document.getElementById('upload-button').addEventListener('click', function() {
    console.log('Upload button clicked');
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    if (file) {
        console.log('File selected:', file.name);
    } else {
        console.log('No file selected');
    }
});