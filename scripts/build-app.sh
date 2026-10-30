#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/app/build/Stitch.app"
SRC="$ROOT/app/Stitch"

if [ ! -x "$SRC/Resources/engine/stitch-engine" ]; then
  "$ROOT/scripts/sync-engine.sh"
fi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

xcrun swiftc -parse-as-library \
  "$SRC/AppKitMain.swift" \
  "$SRC/Models.swift" \
  "$SRC/EngineClient.swift" \
  -o "$APP/Contents/MacOS/Stitch" \
  -framework AppKit \
  -framework UniformTypeIdentifiers \
  -target arm64-apple-macosx14.0 \
  -O

cat > "$APP/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleExecutable</key>
	<string>Stitch</string>
	<key>CFBundleIdentifier</key>
	<string>com.mike.stitch</string>
	<key>CFBundleName</key>
	<string>Stitch</string>
	<key>CFBundleDisplayName</key>
	<string>Stitch</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>0.1.0</string>
	<key>CFBundleVersion</key>
	<string>1</string>
	<key>LSMinimumSystemVersion</key>
	<string>14.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>NSPrincipalClass</key>
	<string>NSApplication</string>
	<key>CFBundleDocumentTypes</key>
	<array>
		<dict>
			<key>CFBundleTypeName</key>
			<string>TIFF Image</string>
			<key>CFBundleTypeRole</key>
			<string>Viewer</string>
			<key>LSHandlerRank</key>
			<string>Default</string>
			<key>CFBundleTypeExtensions</key>
			<array>
				<string>tif</string>
				<string>tiff</string>
				<string>TIF</string>
				<string>TIFF</string>
			</array>
		</dict>
	</array>
</dict>
</plist>
PLIST

cp -R "$SRC/Resources/engine" "$APP/Contents/Resources/engine"
chmod +x "$APP/Contents/Resources/engine/stitch-engine" "$APP/Contents/MacOS/Stitch"
printf 'APPL????' > "$APP/Contents/PkgInfo"
codesign --force --deep -s - "$APP" >/dev/null

echo "Built $APP"
echo "Open with: open \"$APP\""
