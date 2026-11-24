#include "MainWindow.h"
#include <QApplication>
#include <QWebEngineSettings>
#include <QSurfaceFormat>
#include <QQuickWindow>
#include <QSGRendererInterface>
#include <QDir>
#include <QStandardPaths>

int main(int argc, char *argv[])
{
    // Set up OpenGL context for WebGL compatibility
    QSurfaceFormat format;
    format.setVersion(4, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(4);
    QSurfaceFormat::setDefaultFormat(format);
    
    // Set graphics API for Qt Quick (affects WebEngine rendering)
    QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);
    
    QApplication app(argc, argv);
    
    // Set application properties
    app.setApplicationName("NextVerse Browser Platform");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("NextVerse");
    app.setOrganizationDomain("nextverse.local");
    
    // Configure global WebEngine settings
    QWebEngineSettings::defaultSettings()->setAttribute(
        QWebEngineSettings::LocalContentCanAccessFileUrls, true
    );
    QWebEngineSettings::defaultSettings()->setAttribute(
        QWebEngineSettings::LocalContentCanAccessRemoteUrls, true
    );
    QWebEngineSettings::defaultSettings()->setAttribute(
        QWebEngineSettings::AllowRunningInsecureContent, true
    );
    
    // Enable hardware acceleration and WebGL
    QWebEngineSettings::defaultSettings()->setAttribute(
        QWebEngineSettings::WebGLEnabled, true
    );
    QWebEngineSettings::defaultSettings()->setAttribute(
        QWebEngineSettings::Accelerated2dCanvasEnabled, true
    );
    
    // Create and show main window
    MainWindow window;
    window.show();
    
    return app.exec();
}
