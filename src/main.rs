use image::*;
use imageproc::drawing::*;
use imageproc::rect::Rect;
use mtcnn_rs::trt_mtcnn::*;
use show_image::*;
use std::time::Duration;
use tensorrt_rs::runtime::*;

fn main() {
    let logger = Logger::new();
    let mt = mtcnn::new(&logger).unwrap();

    let img = image::open("DSC_0003.JPG").unwrap();

    // let window = make_window("image").unwrap();
    // window.set_image(&img, "image-001").unwrap();

    // while let Ok(event) = window.wait_key(Duration::from_millis(100)) {
    //     if let Some(event) = event {
    //         if event.key == KeyCode::Escape {
    //             break;
    //         }
    //     }
    // }
    println!("start");
    let dets = mt.detect(&img, 40);
    println!("end");

    println!("{:?}", dets);

    let rect = Rect::at(dets[0][0] as i32, dets[0][1] as i32).of_size(
        (dets[0][2] - dets[0][0]) as u32,
        (dets[0][3] - dets[0][1]) as u32,
    );

    let rect1 = Rect::at(dets[0][0] as i32 + 1, dets[0][1] as i32 + 1).of_size(
        (dets[0][2] - dets[0][0]) as u32 - 1,
        (dets[0][3] - dets[0][1]) as u32 - 1,
    );

    let rect2 = Rect::at(dets[0][0] as i32 - 1, dets[0][1] as i32 - 1).of_size(
        (dets[0][2] - dets[0][0]) as u32 + 1,
        (dets[0][3] - dets[0][1]) as u32 + 1,
    );

    println!("{:?}", rect);

    let tagged_img1 = draw_hollow_rect(&img, rect, Rgba([0, 255, 0, 255]));
    let tagged_img2 = draw_hollow_rect(&tagged_img1, rect1, Rgba([0, 255, 0, 255]));
    let tagged_img = draw_hollow_rect(&tagged_img2, rect2, Rgba([0, 255, 0, 255]));

    let options = WindowOptions {
        name: "image".to_string(),
        size: [1920, 1080],
        resizable: true,
        preserve_aspect_ratio: true,
    };
    let window = make_window_full(options).unwrap();
    window.set_image(&tagged_img, "image-001").unwrap();

    while let Ok(event) = window.wait_key(Duration::from_millis(100)) {
        if let Some(event) = event {
            if event.key == KeyCode::Escape {
                break;
            }
        }
    }

    tagged_img.save("face_detect.jpg").unwrap();

    println!("Hello, world!");
}
