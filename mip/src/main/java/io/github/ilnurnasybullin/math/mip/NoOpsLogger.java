package io.github.ilnurnasybullin.math.mip;

import java.util.ResourceBundle;

class NoOpsLogger implements System.Logger {
    @Override
    public String getName() {
        return "NoOpsLogger";
    }

    @Override
    public boolean isLoggable(Level level) {
        return false;
    }

    @Override
    public void log(Level level, ResourceBundle bundle, String msg, Throwable thrown) {}

    @Override
    public void log(Level level, ResourceBundle bundle, String format, Object... params) {}
}
