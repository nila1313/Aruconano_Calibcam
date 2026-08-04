#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>

#include "aruco_nano.h"

namespace fs = std::filesystem;


bool isImageFile(const fs::path& path) {
    std::string extension = path.extension().string();

    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        }
    );

    return (
        extension == ".jpg"
        || extension == ".jpeg"
        || extension == ".png"
        || extension == ".bmp"
        || extension == ".tif"
        || extension == ".tiff"
    );
}


std::string csvEscape(const std::string& value) {
    std::string escaped = "\"";

    for (const char character : value) {
        if (character == '"') {
            escaped += "\"\"";
        } else {
            escaped += character;
        }
    }

    escaped += "\"";
    return escaped;
}


int parseDictionaryId(const std::string& value) {
    std::size_t parsed_characters = 0;
    int dictionary_id = 0;

    try {
        dictionary_id = std::stoi(
            value,
            &parsed_characters
        );
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Dictionary ID must be an integer."
        );
    }

    if (parsed_characters != value.size()) {
        throw std::runtime_error(
            "Dictionary ID contains invalid characters."
        );
    }

    // OpenCV predefined dictionary identifiers currently occupy
    // the integer range 0 through 21.
    if (dictionary_id < 0 || dictionary_id > 21) {
        throw std::runtime_error(
            "Dictionary ID must be between 0 and 21."
        );
    }

    return dictionary_id;
}


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "Usage:\n"
            << "  aruco_nano_detector "
            << "<input_folder> <output_folder> <dictionary_id>\n\n"
            << "Example for DICT_6X6_250:\n"
            << "  aruco_nano_detector frames output 10\n";

        return 1;
    }

    const fs::path input_folder = fs::absolute(argv[1]);
    const fs::path output_folder = fs::absolute(argv[2]);

    int dictionary_id = 0;

    try {
        dictionary_id = parseDictionaryId(argv[3]);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }

    if (
        !fs::exists(input_folder)
        || !fs::is_directory(input_folder)
    ) {
        std::cerr
            << "ERROR: Input folder does not exist: "
            << input_folder
            << "\n";

        return 1;
    }

    if (fs::exists(output_folder)) {
        std::cerr
            << "ERROR: Output path already exists: "
            << output_folder
            << "\n";

        return 1;
    }

    std::vector<fs::path> image_files;

    for (
        const auto& entry :
        fs::directory_iterator(input_folder)
    ) {
        if (
            entry.is_regular_file()
            && isImageFile(entry.path())
        ) {
            image_files.push_back(entry.path());
        }
    }

    std::sort(
        image_files.begin(),
        image_files.end()
    );

    if (image_files.empty()) {
        std::cerr
            << "ERROR: No supported image files were found in: "
            << input_folder
            << "\n";

        return 1;
    }

    fs::create_directories(output_folder);

    const fs::path detections_path =
        output_folder / "detections.csv";

    const fs::path summary_path =
        output_folder / "frame_summary.csv";

    std::ofstream detections_csv(detections_path);
    std::ofstream summary_csv(summary_path);

    if (
        !detections_csv.is_open()
        || !summary_csv.is_open()
    ) {
        std::cerr
            << "ERROR: Could not create detector CSV files.\n";

        return 1;
    }

    detections_csv
        << "frame,marker_id,"
        << "corner0_x,corner0_y,"
        << "corner1_x,corner1_y,"
        << "corner2_x,corner2_y,"
        << "corner3_x,corner3_y,"
        << "center_x,center_y\n";

    summary_csv
        << "frame,num_markers\n";

    // Keep native detection deterministic and independent of
    // machine-specific OpenCL or thread scheduling behaviour.
    cv::ocl::setUseOpenCL(false);
    cv::setNumThreads(1);

    const cv::aruco::Dictionary dictionary =
        cv::aruco::getPredefinedDictionary(
            static_cast<
                cv::aruco::PredefinedDictionaryType
            >(dictionary_id)
        );

    aruco_nano::ArucoDetector detector(dictionary);

    std::size_t total_markers = 0;

    detections_csv
        << std::setprecision(9);

    std::cout
        << "ArUco Nano detector\n"
        << "===================\n"
        << "Input:         " << input_folder << "\n"
        << "Output:        " << output_folder << "\n"
        << "Dictionary ID: " << dictionary_id << "\n"
        << "Images:        " << image_files.size() << "\n"
        << "OpenCL:        OFF\n"
        << "OpenCV threads: 1\n\n";

    for (
        std::size_t image_index = 0;
        image_index < image_files.size();
        ++image_index
    ) {
        const fs::path& image_path =
            image_files[image_index];

        const cv::Mat image = cv::imread(
            image_path.string(),
            cv::IMREAD_COLOR
        );

        if (image.empty()) {
            std::cerr
                << "ERROR: Could not decode image: "
                << image_path
                << "\n";

            return 2;
        }

        std::vector<int> marker_ids;
        std::vector<std::vector<cv::Point2f>>
            marker_corners;

        detector.detectMarkers(
            image,
            marker_corners,
            marker_ids
        );

        if (
            marker_ids.size()
            != marker_corners.size()
        ) {
            std::cerr
                << "ERROR: Marker ID and corner counts differ for "
                << image_path.filename()
                << "\n";

            return 2;
        }

        const std::string filename =
            image_path.filename().string();

        summary_csv
            << csvEscape(filename)
            << ","
            << marker_ids.size()
            << "\n";

        total_markers += marker_ids.size();

        for (
            std::size_t marker_index = 0;
            marker_index < marker_ids.size();
            ++marker_index
        ) {
            const auto& corners =
                marker_corners[marker_index];

            if (corners.size() != 4) {
                std::cerr
                    << "ERROR: Marker "
                    << marker_ids[marker_index]
                    << " does not contain four corners in "
                    << filename
                    << "\n";

                return 2;
            }

            cv::Point2f center(0.0F, 0.0F);

            for (const cv::Point2f& corner : corners) {
                center += corner;
            }

            center.x /= 4.0F;
            center.y /= 4.0F;

            detections_csv
                << csvEscape(filename)
                << ","
                << marker_ids[marker_index]
                << ","
                << corners[0].x << ","
                << corners[0].y << ","
                << corners[1].x << ","
                << corners[1].y << ","
                << corners[2].x << ","
                << corners[2].y << ","
                << corners[3].x << ","
                << corners[3].y << ","
                << center.x << ","
                << center.y << "\n";
        }

        if (
            (image_index + 1) % 20 == 0
            || image_index + 1 == image_files.size()
        ) {
            std::cout
                << "Processed "
                << image_index + 1
                << " / "
                << image_files.size()
                << " images; total markers: "
                << total_markers
                << "\n";
        }
    }

    detections_csv.close();
    summary_csv.close();

    std::cout
        << "\nDetection completed.\n"
        << "Frames processed: "
        << image_files.size()
        << "\n"
        << "Markers detected: "
        << total_markers
        << "\n"
        << "Detections: "
        << detections_path
        << "\n"
        << "Summary: "
        << summary_path
        << "\n";

    return 0;
}
