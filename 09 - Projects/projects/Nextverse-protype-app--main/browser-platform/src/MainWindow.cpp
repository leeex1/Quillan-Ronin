#include "MainWindow.h"
#include "LocalProxy.h"
#include "WebBridge.h"
#include <QApplication>
#include <QWebEngineSettings>
#include <QWebEngineProfile>
#include <QWebEngineView>
#include <QWebChannel>
#include <QVBoxLayout>
#include <QMenuBar>
#include <QStatusBar>
#include <QAction>
#include <QKeySequence>
#include <QMessageBox>
#include <QStandardPaths>
#include <QDir>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QTimer>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_webView(nullptr)
    , m_webChannel(nullptr)
    , m_localProxy(nullptr)
    , m_webBridge(nullptr)
    , m_developmentMode(true) // Set to false for production builds
    , m_reactDevUrl("http://localhost:3000")
    , m_productionPath("qrc:/web/index.html")
{
    setupUI();
    setupWebEngine();
    setupMenus();
    setupWebChannel();
    loadReactApp();
    
    // Window properties
    setWindowTitle("NextVerse Browser Platform");
    setMinimumSize(1200, 800);
    resize(1600, 1000);
    
    // Center window on screen
    QRect screenGeometry = QApplication::primaryScreen()->geometry();
    int x = (screenGeometry.width() - width()) / 2;
    int y = (screenGeometry.height() - height()) / 2;
    move(x, y);
}

MainWindow::~MainWindow()
{
    if (m_webChannel) {
        m_webChannel->deleteLater();
    }
}

void MainWindow::setupUI()
{
    // Create central widget and layout
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    
    // Create web view
    m_webView = new QWebEngineView(this);
    layout->addWidget(m_webView);
    
    // Connect signals
    connect(m_webView, &QWebEngineView::loadFinished, this, &MainWindow::onLoadFinished);
    connect(m_webView, &QWebEngineView::loadStarted, this, &MainWindow::onLoadStarted);
    connect(m_webView, &QWebEngineView::urlChanged, this, &MainWindow::onUrlChanged);
    
    // Status bar
    statusBar()->showMessage("Ready");
}

void MainWindow::setupWebEngine()
{
    // Configure WebEngine settings for local development
    QWebEngineSettings *settings = m_webView->settings();
    
    // Enable local file access and other necessary permissions
    settings->setAttribute(QWebEngineSettings::LocalContentCanAccessFileUrls, true);
    settings->setAttribute(QWebEngineSettings::LocalContentCanAccessRemoteUrls, true);
    settings->setAttribute(QWebEngineSettings::AllowRunningInsecureContent, true);
    settings->setAttribute(QWebEngineSettings::JavascriptEnabled, true);
    settings->setAttribute(QWebEngineSettings::LocalStorageEnabled, true);
    settings->setAttribute(QWebEngineSettings::WebGLEnabled, true);
    settings->setAttribute(QWebEngineSettings::Accelerated2dCanvasEnabled, true);
    settings->setAttribute(QWebEngineSettings::AutoLoadImages, true);
    settings->setAttribute(QWebEngineSettings::PluginsEnabled, true);
    
    // Configure profile for optimal performance
    QWebEngineProfile *profile = m_webView->page()->profile();
    profile->setHttpCacheType(QWebEngineProfile::MemoryHttpCache);
    profile->setHttpCacheMaximumSize(50 * 1024 * 1024); // 50MB cache
    
    // Set persistent storage path for local data
    QString dataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dataPath);
    profile->setPersistentStoragePath(dataPath + "/storage");
}

void MainWindow::setupMenus()
{
    // File menu
    QMenu *fileMenu = menuBar()->addMenu("&File");
    
    QAction *reloadAction = new QAction("&Reload", this);
    reloadAction->setShortcut(QKeySequence::Refresh);
    connect(reloadAction, &QAction::triggered, this, &MainWindow::reloadPage);
    fileMenu->addAction(reloadAction);
    
    fileMenu->addSeparator();
    
    QAction *exitAction = new QAction("E&xit", this);
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    fileMenu->addAction(exitAction);
    
    // View menu
    QMenu *viewMenu = menuBar()->addMenu("&View");
    
    QAction *devToolsAction = new QAction("&Developer Tools", this);
    devToolsAction->setShortcut(QKeySequence("F12"));
    connect(devToolsAction, &QAction::triggered, this, &MainWindow::toggleDevTools);
    viewMenu->addAction(devToolsAction);
    
    // Help menu
    QMenu *helpMenu = menuBar()->addMenu("&Help");
    
    QAction *aboutAction = new QAction("&About", this);
    connect(aboutAction, &QAction::triggered, [this]() {
        QMessageBox::about(this, "About NextVerse Browser Platform",
            "NextVerse Browser Platform v1.0.0\n\n"
            "A hybrid Qt+React application leveraging QWebEngineView\n"
            "for high-performance local platform development.\n\n"
            "Built with Qt6 WebEngine and React frontend.");
    });
    helpMenu->addAction(aboutAction);
}

void MainWindow::setupWebChannel()
{
    // Create web channel for Qt <-> JavaScript communication
    m_webChannel = new QWebChannel(this);
    
    // Create and register proxy objects
    m_localProxy = new LocalProxy(this);
    m_webBridge = new WebBridge(this);
    
    m_webChannel->registerObject("localProxy", m_localProxy);
    m_webChannel->registerObject("webBridge", m_webBridge);
    
    // Set the web channel on the page
    m_webView->page()->setWebChannel(m_webChannel);
}

void MainWindow::loadReactApp()
{
    if (m_developmentMode) {
        // Development mode: Load from localhost:3000
        statusBar()->showMessage("Loading React development server...");
        m_webView->load(QUrl(m_reactDevUrl));
    } else {
        // Production mode: Load from embedded resources
        statusBar()->showMessage("Loading production build...");
        m_webView->load(QUrl(m_productionPath));
    }
}

void MainWindow::onLoadFinished(bool success)
{
    if (success) {
        statusBar()->showMessage("Ready", 2000);
    } else {
        statusBar()->showMessage("Failed to load page", 5000);
        
        if (m_developmentMode) {
            // If development server is not running, show helpful message
            QMessageBox::warning(this, "Development Server Not Found",
                "Could not connect to React development server at localhost:3000.\n\n"
                "Please ensure you have started the React development server:\n"
                "cd react-frontend && npm run dev");
        }
    }
}

void MainWindow::onLoadStarted()
{
    statusBar()->showMessage("Loading...");
}

void MainWindow::onUrlChanged(const QUrl &url)
{
    // Update window title with current URL for development
    if (m_developmentMode) {
        setWindowTitle(QString("NextVerse Browser Platform - %1").arg(url.toString()));
    }
}

void MainWindow::toggleDevTools()
{
    // Toggle developer tools
    if (m_webView->page()->devToolsPage()) {
        m_webView->page()->setDevToolsPage(nullptr);
    } else {
        QWebEngineView *devView = new QWebEngineView();
        devView->setAttribute(Qt::WA_DeleteOnClose);
        devView->resize(1200, 600);
        devView->show();
        m_webView->page()->setDevToolsPage(devView->page());
    }
}

void MainWindow::reloadPage()
{
    m_webView->reload();
}
