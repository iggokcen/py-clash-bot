using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace PyClashBotLauncher
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string pythonwPath = Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe");
                string pythonPath = Path.Combine(baseDir, ".venv", "Scripts", "python.exe");

                ProcessStartInfo psi = new ProcessStartInfo();
                psi.WorkingDirectory = baseDir;

                if (File.Exists(pythonwPath))
                {
                    psi.FileName = pythonwPath;
                    psi.Arguments = "-m pyclashbot";
                    psi.UseShellExecute = false;
                    psi.CreateNoWindow = true;
                }
                else if (File.Exists(pythonPath))
                {
                    psi.FileName = pythonPath;
                    psi.Arguments = "-m pyclashbot";
                    psi.UseShellExecute = false;
                }
                else
                {
                    psi.FileName = "uv";
                    psi.Arguments = "run python pyclashbot";
                    psi.UseShellExecute = true;
                }

                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Py-Clash-Bot başlatılırken hata oluştu:\n\n" + ex.Message,
                    "PyClashBot Başlatıcı",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }
    }
}
