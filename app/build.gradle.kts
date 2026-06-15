plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.graduationprojectsapp"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.graduationprojectsapp"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            // arm64-v8a: all modern Android phones
            // armeabi-v7a: older 32-bit devices
            // x86_64: emulators
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildFeatures {
        viewBinding = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            excludes += "META-INF/DEPENDENCIES"
        }
    }
}

dependencies {
    // ── AndroidX core ─────────────────────────────────────────────────────────
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation("androidx.cardview:cardview:1.0.0")

    // ── CameraX ───────────────────────────────────────────────────────────────
    implementation("androidx.camera:camera-core:1.5.1")
    implementation("androidx.camera:camera-camera2:1.5.1")
    implementation("androidx.camera:camera-lifecycle:1.5.1")
    implementation("androidx.camera:camera-view:1.5.1")

    // ── TensorFlow Lite ───────────────────────────────────────────────────────
    // Core runtime — required by both YOLOv11Detector and YOLOv26Detector
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    // Support library — required by YOLOv11Detector (TensorImage, NormalizeOp, etc.)
    implementation("org.tensorflow:tensorflow-lite-support:0.5.0")
    // GPU delegate — enables YOLOv26Detector's optional useGpu=true mode
    // Falls back to CPU automatically if GPU is unavailable (handled in YOLOv26Detector.init)
    implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")

    // ── ONNX Runtime — Depth Anything V2 ─────────────────────────────────────
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")

    // ── USB Serial — ESP32 communication ─────────────────────────────────────
    implementation("com.github.mik3y:usb-serial-for-android:3.9.0")

    // ── Test ──────────────────────────────────────────────────────────────────
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
