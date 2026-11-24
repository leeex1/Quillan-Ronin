#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWebEngineView>
#include <QWebChannel>
#include <QVBoxLayout>
#include <QMenuBar>
#include <QStatusBar>
#include <QUrl>
#include <QWebEngineSettings>
#include <QWebEngineProfile>
#include <QApplication>

class LocalProxy;
class WebBridge;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onLoadFinished(bool success);
    void onLoadStarted();
    void onUrlChanged(const QUrl &url);
    void toggleDevTools();
    void reloadPage();

private:
    void setupUI();
    void setupWebEngine();
    void setupMenus();
    void setupWebChannel();
    void loadReactApp();
    
    QWebEngineView *m_webView;
    QWebChannel *m_webChannel;
    LocalProxy *m_localProxy;
    WebBridge *m_webBridge;
    
    // Development mode flag
    bool m_developmentMode;
    QString m_reactDevUrl;
    QString m_productionPath;
};

#endif // MAINWINDOW_H
